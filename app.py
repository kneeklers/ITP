from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Use 0 for default camera, adjust if needed
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

def generate_frames():
    while True:
        with camera_lock:
            camera = get_camera()
            success, frame = camera.read()
            if not success:
                break
            else:
                # Process frame for defect detection
                processed_frame = detect_defects(frame)
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.1)  # Add small delay to prevent overwhelming the system

@app.route('/')
def index():
    return render_template('index.html')

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