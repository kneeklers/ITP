from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import threading
import time
import os
from trt_infer import TensorRTInference

app = Flask(__name__)

# Global variables (except for 'camera' object itself, now managed by CameraStream)
# camera = None # Removed global camera object
camera_lock = threading.Lock() # Still used by get_camera for thread-safety during creation
model = None
model_lock = threading.Lock()

# GStreamer pipeline for camera
GSTREAMER_PIPELINE = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false max-buffers=1 wait-on-eos=false"

# Load the model once at startup for efficiency
MODEL_PATH = 'models/your_model.engine'

# Create uploads and results directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variable to hold the latest frame from the camera stream
latest_frame = None
latest_frame_lock = threading.Lock()
camera_stream_running = False

class CameraStream:
    def __init__(self):
        self._camera_instance = None # Private instance of VideoCapture
        self.running = False
        self.thread = None
        self.lock = threading.Lock() # For protecting internal state of CameraStream

    def start(self):
        global camera_stream_running
        with self.lock:
            if self.running:
                print("CameraStream: Already running.")
                return

            self.running = True
            camera_stream_running = True
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True # Daemon threads exit when the main program exits
            self.thread.start()
            print("CameraStream thread started.")

    def _update(self):
        global latest_frame
        global camera_stream_running

        print("CameraStream._update: Thread started, acquiring camera...")
        # Acquire a new camera instance for this thread
        self._camera_instance = _get_camera_instance()
        if self._camera_instance is None:
            print("CameraStream._update: Failed to acquire camera, thread stopping.")
            self.running = False
            camera_stream_running = False
            return

        print("CameraStream._update: Camera acquired. Starting frame read loop.")

        frame_read_count = 0
        consecutive_read_failures = 0
        max_read_failures = 10 # Allow more failures before full reinit

        while self.running:
            print("CameraStream._update: Attempting to read frame from camera...")
            ret, frame = self._camera_instance.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                print(f"CameraStream._update: Failed to read frame ({consecutive_read_failures}/{max_read_failures}). Ret: {ret}, Frame is None: {frame is None}")
                if consecutive_read_failures >= max_read_failures:
                    print("CameraStream._update: Max consecutive frame read failures reached. Releasing and re-acquiring camera.")
                    if self._camera_instance:
                        self._camera_instance.release()
                    self._camera_instance = _get_camera_instance() # Re-acquire camera
                    if self._camera_instance is None:
                        print("CameraStream._update: Re-acquisition failed, thread stopping.")
                        self.running = False
                        camera_stream_running = False
                        break
                    consecutive_read_failures = 0 # Reset failures on successful re-acquisition
                    time.sleep(0.5) # Small delay after re-acquisition
                time.sleep(0.05) # Small delay on read failure to prevent busy-waiting
                continue

            consecutive_read_failures = 0  # Reset on successful frame read
            frame_read_count += 1
            if frame_read_count % 100 == 0:
                print(f"CameraStream._update: Successfully read {frame_read_count} frames.")

            with latest_frame_lock:
                latest_frame = frame.copy()

            time.sleep(0.01) # Small delay to control CPU usage

        print("CameraStream._update: Thread stopping. Releasing camera instance.")
        if self._camera_instance:
            self._camera_instance.release()
        self._camera_instance = None
        self.running = False
        camera_stream_running = False

    def stop(self):
        with self.lock:
            if not self.running:
                print("CameraStream: Not running.")
                return
            self.running = False
            if self.thread and self.thread.is_alive():
                print("CameraStream: Waiting for thread to finish...")
                self.thread.join(timeout=5) # Increased timeout for robustness
                if self.thread.is_alive():
                    print("CameraStream: Warning! Thread did not terminate gracefully within timeout.")
                else:
                    print("CameraStream: Thread terminated successfully.")
            self.thread = None
            print("CameraStream thread stopped.")

# Instantiate the camera stream manager
camera_manager = CameraStream()

# Renamed get_camera to _get_camera_instance to emphasize it returns a new instance
def _get_camera_instance():
    # This function now exclusively focuses on creating and testing a new VideoCapture instance.
    # It does not manage a global 'camera' variable.
    with camera_lock: # Still use the lock to prevent multiple threads from initializing cameras concurrently
        print("Attempting to create and test a new camera instance...")
        if GSTREAMER_PIPELINE is None:
            print("ERROR: GSTREAMER_PIPELINE is not set. Please uncomment and configure it in app.py")
            return None

        try:
            print(f"Opening camera with GStreamer pipeline: {GSTREAMER_PIPELINE}")
            cam_instance = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

            if cam_instance.isOpened():
                test_frames_to_read = 5
                frames_read_successfully = 0
                for i in range(test_frames_to_read):
                    ret, frame = cam_instance.read()
                    if ret and frame is not None:
                        frames_read_successfully += 1
                        time.sleep(0.01)
                    else:
                        print(f"Failed to read test frame {i+1}/{test_frames_to_read} from GStreamer pipeline. Ret: {ret}, Frame is None: {frame is None}")

                if frames_read_successfully >= 3:
                    print(f"Camera instance successfully created and tested. Read {frames_read_successfully} test frames.")
                    return cam_instance # Return the newly created and tested instance
                else:
                    print("Not enough test frames could be read. Releasing new camera instance.")
                    cam_instance.release()
                    return None
            else:
                print("Camera instance not opened with GStreamer pipeline.")
                return None

        except Exception as e:
            print(f"Error during GStreamer camera instance creation: {e}")
            # Ensure the instance is released if an error occurs during its creation
            if 'cam_instance' in locals() and cam_instance.isOpened():
                cam_instance.release()
            return None

def release_camera():
    # This function now primarily stops the CameraStream thread.
    # The CameraStream thread is responsible for releasing its own VideoCapture instance.
    global camera_stream_running
    if camera_stream_running:
        print("release_camera: Stopping CameraStream thread.")
        camera_manager.stop()
        # Clear latest_frame to prevent stale data and ensure resource release
        global latest_frame
        with latest_frame_lock:
            latest_frame = None
    else:
        print("release_camera: CameraStream not running or already stopped.")

def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                # Ensure any existing CUDA context is cleaned up
                try:
                    import pycuda.driver as cuda
                    cuda.Context.pop()
                except:
                    pass
                
                model = TensorRTInference(MODEL_PATH)
                print("Model initialized successfully")
            except Exception as e:
                print(f"Error initializing model: {e}")
                return None
    return model

def release_model():
    global model
    with model_lock:
        if model is not None:
            try:
                print("release_model: Attempting to destroy model resources.")
                model.destroy() # Explicitly call the destroy method
                model = None
                print("release_model: Model released successfully.")
            except Exception as e:
                print(f"release_model: Error destroying or releasing model: {e}")
                model = None # Ensure model is cleared even if destroy fails partially

def detect_defects(frame):
    # For now, just return the original frame
    # You can add your defect detection logic here later
    return frame

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image:", image_path)
            return None

        # Get model instance
        model_instance = get_model()
        if model_instance is None:
            return {"error": "Failed to initialize model"}

        # Run inference and visualization
        result_image, boxes, scores, class_ids = model_instance.infer_and_visualize(
            image,
            conf_threshold=0.3,
            nms_threshold=0.4,
            save_path=None
        )

        # Save the result image to static/results for web access
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join('static/results', result_filename)
        cv2.imwrite(result_path, result_image)

        # Prepare summary for the web page
        summary = {
            'defect_type': ', '.join([model_instance.class_names.get(c, str(c)) for c in class_ids]) if class_ids else 'None',
            'confidence': ', '.join([f'{s*100:.2f}' for s in scores]) if scores else '0.00',
            'result_image': result_filename
        }
        return summary

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def generate_frames():
    global latest_frame
    print("generate_frames: Starting frame generation loop (reading from shared buffer)...")

    while True:
        try:
            # Ensure the CameraStream thread is running to provide frames
            if not camera_manager.running:
                print("generate_frames: CameraStream is not running. Attempting to start.")
                camera_manager.start()
                time.sleep(1) # Give time for the stream to start

            # Wait for the first frame to become available
            if latest_frame is None:
                print("generate_frames: No frames in buffer yet. Waiting...")
                time.sleep(0.5) # Wait for a frame to become available
                continue

            with latest_frame_lock: # Acquire lock to safely read the frame
                frame_to_process = latest_frame.copy()
            
            # Process frame
            processed_frame = detect_defects(frame_to_process)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                print("generate_frames: Failed to encode frame.")
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033) # ~30 FPS

        except Exception as e:
            print(f"generate_frames: Error in loop: {e}")
            time.sleep(1) # Wait before retrying
            continue

    print("generate_frames: Exiting loop.")

@app.route('/')
def index():
    try:
        print("Accessing index route")
        release_model() # Release model if moving to live stream
        camera_manager.start() # Ensure camera stream is running
        time.sleep(0.5) # Give the camera stream a moment to start
        return render_template('index.html')
    except Exception as e:
        print(f"Error in index route: {e}")
        return "Error initializing camera", 500

@app.route('/video_feed')
def video_feed():
    try:
        print("Starting video feed")
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed route: {e}")
        return "Error streaming video", 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    try:
        print("Accessing upload page")
        camera_manager.stop() # Stop camera stream for upload
        time.sleep(0.75) # Add a short delay to allow camera resources to fully release
        release_model() # Ensure model is released before processing

        if request.method == 'POST':
            try:
                if 'image' not in request.files:
                    print("No image file in request")
                    return render_template('upload.html', result=None, error="No image file selected")
                
                file = request.files['image']
                if file.filename == '':
                    print("No selected file")
                    return render_template('upload.html', result=None, error="No file selected")
                
                if not file:
                    print("Invalid file")
                    return render_template('upload.html', result=None, error="Invalid file")
                
                os.makedirs('uploads', exist_ok=True)
                filename = os.path.join('uploads', file.filename)
                file.save(filename)
                print(f"File saved to {filename}")
                
                result = process_image(filename)
                print(f"Processing result: {result}")
                
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Error removing file: {e}")
                
                if result is None:
                    return render_template('upload.html', result=None, error="Error processing image")
                
                return render_template('upload.html', result=result)
                
            except Exception as e:
                print(f"Error in upload POST: {str(e)}")
                return render_template('upload.html', result=None, error=f"Error: {str(e)}")
        
        return render_template('upload.html', result=None)
    except Exception as e:
        print(f"Error in upload route: {e}")
        return "Error loading upload page", 500

@app.route('/shutdown')
def shutdown():
    try:
        print("Shutting down application...")
        release_camera() # This will stop the CameraStream thread
        release_model()
        return "Resources released"
    except Exception as e:
        print(f"Error in shutdown: {e}")
        return "Error releasing resources", 500

if __name__ == '__main__':
    try:
        print("Application starting...")
        release_camera() # Ensure camera manager is stopped at startup
        release_model()
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=False) # threaded=False as we manage our own threads
    finally:
        print("Application exiting. Ensuring resources are released.")
        release_camera()
        release_model() 