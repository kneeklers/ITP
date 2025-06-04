from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import threading
import time
import os
from trt_infer import TensorRTInference

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()

# Load the model once at startup for efficiency
MODEL_PATH = 'models/your_model.engine'
model = TensorRTInference(MODEL_PATH)

# Create uploads and results directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

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
    # Placeholder for live defect detection
    return frame

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image:", image_path)
            return None

        # Run inference and visualization
        result_image, boxes, scores, class_ids = model.infer_and_visualize(
            image,
            conf_threshold=0.3,  # adjust as needed
            nms_threshold=0.4,
            save_path=None
        )

        # Save the result image to static/results for web access
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join('static/results', result_filename)
        cv2.imwrite(result_path, result_image)

        # Prepare summary for the web page
        summary = {
            'defect_type': ', '.join([model.class_names.get(c, str(c)) for c in class_ids]) if class_ids else 'None',
            'confidence': ', '.join([f'{s*100:.2f}' for s in scores]) if scores else '0.00',
            'result_image': result_filename
        }
        return summary

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
        file.save(filename)
        result = process_image(filename)
        os.remove(filename)
        return render_template('upload.html', result=result)
    print("Attempting to load upload page")
    return "Upload page reached!"

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
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
    finally:
        release_camera() 