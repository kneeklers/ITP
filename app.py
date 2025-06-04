from flask import Flask, render_template, Response, request, redirect, url_for
from flask_socketio import SocketIO
import cv2
import numpy as np
import threading
import time
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

app = Flask(__name__)
socketio = SocketIO(app)

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
        # Load the TensorRT engine
        engine = load_engine('models/your_model.engine')
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Adjust size based on your model's requirements
        img = img.astype(np.float32) / 255.0  # Normalize
        
        # Prepare input and output buffers
        input_shape = (1, 3, 224, 224)  # Adjust based on your model
        output_shape = (1, num_classes)  # Adjust based on your model
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(1 * img.nbytes)
        d_output = cuda.mem_alloc(1 * output_shape[0] * np.dtype(np.float32).itemsize)
        
        # Copy input to GPU
        cuda.memcpy_htod(d_input, img)
        
        # Run inference
        context.execute_v2(bindings=[int(d_input), int(d_output)])
        
        # Get results
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        # Process results
        defect_type = np.argmax(output[0])
        confidence = float(output[0][defect_type] * 100)
        
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

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        filename = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filename)
        
        # Process the image
        result = process_image(filename)
        
        # Clean up
        os.remove(filename)
        
        return render_template('upload.html', result=result)

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
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    finally:
        release_camera() 