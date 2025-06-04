from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import threading
import time
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def detect_defects(frame):
    # TODO: Implement your defect detection algorithm here
    # This is a placeholder that just returns the original frame
    return frame

def process_image(image_path):
    try:
        engine = load_engine('models/your_model.engine')
        context = engine.create_execution_context()

        img = cv2.imread(image_path)
        img = cv2.resize(img, (832, 832))
        img = img.astype(np.float32) / 255.0

        input_binding_idx = engine.get_binding_index('images')
        output_binding_idx = engine.get_binding_index('output 0')
        input_shape = engine.get_binding_shape(input_binding_idx)
        output_shape = engine.get_binding_shape(output_binding_idx)

        # Prepare input and output buffers
        input_buffer = np.zeros(input_shape, dtype=np.float32)
        output_buffer = np.zeros(output_shape, dtype=np.float32)
        input_buffer[0] = img.transpose(2, 0, 1)  # HWC to CHW

        # Allocate device memory
        d_input = cuda.mem_alloc(input_buffer.nbytes)
        d_output = cuda.mem_alloc(output_buffer.nbytes)

        # Copy input to device
        cuda.memcpy_htod(d_input, input_buffer)

        # Run inference
        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings=bindings)

        # Copy output from device
        cuda.memcpy_dtoh(output_buffer, d_output)

        defect_type = np.argmax(output_buffer[0])
        confidence = float(output_buffer[0][defect_type] * 100)

        return {
            'defect_type': f'Defect Type {defect_type}',
            'confidence': f'{confidence:.2f}'
        }
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def generate_frames():
    while True:
        with camera_lock:
            camera = get_camera()
            success, frame = camera.read()
            if not success:
                break
            else:
                processed_frame = detect_defects(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('upload.html', result=None)
        file = request.files['image']
        filename = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filename)
        result = process_image(filename)
        os.remove(filename)
        return render_template('upload.html', result=result)
    return render_template('upload.html', result=None)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    release_camera()
    return "Camera released"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        release_camera() 