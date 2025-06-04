from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None:
        # Try different camera indices if 0 doesn't work
        for i in range(2):  # Try camera 0 and 1
            try:
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    # Set lower resolution for better performance
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return camera
            except:
                continue
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
            if camera is None:
                continue
                
            success, frame = camera.read()
            if not success:
                continue
            else:
                # Process frame for defect detection
                processed_frame = detect_defects(frame)
                
                # Convert to JPEG with lower quality for better performance
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
        # Use a simpler server configuration
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        release_camera() 