from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
import threading
import time
import os
from trt_infer import TensorRTInference
import pycuda.driver as cuda
cuda.init()

app = Flask(__name__)

# Global variables (except for 'camera' object itself, now managed by CameraStream)
# camera = None # Removed global camera object
camera_lock = threading.Lock() # Still used by _get_camera_instance for thread-safety during creation
model = None
model_lock = threading.Lock()

# GStreamer pipeline for camera - Now explicitly defined with queue properties
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


# Load the model once at startup for efficiency
MODEL_PATH = 'models/best12.engine'

# Create uploads, results, and original image directories if they don't exist
os.makedirs('uploads', exist_ok=True) # This can probably be removed now if we save directly to static/uploaded_originals
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/uploaded_originals', exist_ok=True)

# Global variable to hold the latest frame from the camera stream
latest_frame = None
latest_frame_lock = threading.Lock()
camera_stream_running = False

# Add global variable for latest detection info
latest_detection_info = {
    'defect_info': 'No defects detected',
    'defect_types': [],
    'confidences': []
}
latest_detection_info_lock = threading.Lock()

# Add global variable for detection history
latest_detection_history = []  # List of dicts: {'type': ..., 'confidence': ..., 'timestamp': ...}

class CameraStream:
    def __init__(self):
        self._camera_instance = None # Private instance of VideoCapture
        self.running = False
        self._stop_event = threading.Event() # Event to signal thread to stop
        self.thread = None
        self.lock = threading.Lock() # For protecting internal state of CameraStream

    def start(self):
        global camera_stream_running
        with self.lock:
            if self.running:
                print("CameraStream: Already running.")
                return

            self.running = True
            self._stop_event.clear() # Clear the stop event
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True # Daemon threads exit when the main program exits
            self.thread.start()
            print("CameraStream thread started.")

    def _update(self):
        global latest_frame
        global camera_stream_running

        print("CameraStream._update: Thread started, acquiring camera...")
        try:
            current_ctx_on_thread = cuda.Context.get_current()
            if current_ctx_on_thread:
                current_ctx_on_thread.pop()
                print("CameraStream._update: Popped existing CUDA context on thread before camera acquisition.")
        except cuda.LogicError:
            pass
        except Exception as e:
            print(f"CameraStream._update: Error popping CUDA context on thread startup: {e}")

        self._camera_instance = _get_camera_instance()
        if self._camera_instance is None:
            print("CameraStream._update: Failed to acquire camera, thread stopping.")
            self.running = False
            camera_stream_running = False
            try:
                current_ctx_on_thread = cuda.Context.get_current()
                if current_ctx_on_thread:
                    current_ctx_on_thread.pop()
            except:
                pass
            return

        print("CameraStream._update: Camera acquired. Starting frame read loop.")

        frame_read_count = 0
        consecutive_read_failures = 0
        max_read_failures = 10

        while self.running:
            # Prioritize stopping immediately if stop event is set
            if self._stop_event.is_set():
                print("CameraStream._update: Stop event set. Exiting loop immediately.")
                break

            # If camera instance becomes None (due to forceful release from stop()), exit
            if self._camera_instance is None:
                print("CameraStream._update: Camera instance is None. Exiting loop.")
                break

            # If camera is no longer open, exit
            if not self._camera_instance.isOpened():
                print("CameraStream._update: Camera is no longer open. Exiting loop.")
                break

            # Add a very small sleep here to yield control, but not too much to affect FPS
            time.sleep(0.001) # This small sleep is crucial for responsiveness to stop signals

            print("CameraStream._update: Attempting to read frame...") # New diagnostic print
            ret, frame = self._camera_instance.read()
            print(f"CameraStream._update: Read result: ret={ret}, frame is None={frame is None}") # New diagnostic print

            # Check stop event immediately after read returns
            if self._stop_event.is_set():
                print("CameraStream._update: Stop event set after reading frame. Exiting loop.")
                break

            if not ret or frame is None:
                consecutive_read_failures += 1
                print(f"CameraStream._update: Failed to read frame ({consecutive_read_failures}/{max_read_failures}). Ret: {ret}, Frame is None: {frame is None}")
                if consecutive_read_failures >= max_read_failures:
                    print("CameraStream._update: Max consecutive frame read failures reached. Terminating thread.")
                    # Force thread to exit if camera is consistently failing
                    self.running = False
                    camera_stream_running = False
                    break # Break the while loop to exit the thread
                time.sleep(0.05)
                continue

            consecutive_read_failures = 0
            frame_read_count += 1
            if frame_read_count % 100 == 0:
                print(f"CameraStream._update: Successfully read {frame_read_count} frames.")

            with latest_frame_lock:
                latest_frame = frame.copy()

            time.sleep(0.01)

        print("CameraStream._update: Thread stopping. Releasing camera instance (if still held).")
        if self._camera_instance:
            self._camera_instance.release()
        self._camera_instance = None
        self.running = False
        camera_stream_running = False
        
        try:
            current_ctx_on_thread = cuda.Context.get_current()
            if current_ctx_on_thread:
                current_ctx_on_thread.pop()
                print("CameraStream._update: Popped CUDA context on thread exit.")
        except cuda.LogicError:
            pass
        except Exception as e:
            print(f"CameraStream._update: Error popping CUDA context on thread exit: {e}")

    def stop(self):
        with self.lock:
            if not self.running:
                print("CameraStream: Not running.")
                return
            
            print("CameraStream: Signalling thread to stop...")
            self._stop_event.set() # Set the stop event first
            self.running = False # Also set this flag for consistency
            
            # Attempt to release camera instance here to unblock read() from main thread
            if self._camera_instance and self._camera_instance.isOpened():
                print("CameraStream: Forcefully releasing internal camera instance from stop() method to unblock thread.")
                self._camera_instance.release()
                self._camera_instance = None # Explicitly set to None here
                print("CameraStream: Internal camera instance release attempted and set to None.")

            # --- Aggressive Camera Reset Attempt (New) ---
            print("CameraStream.stop: Attempting aggressive camera reset...")
            try:
                temp_camera = _get_camera_instance()
                if temp_camera:
                    print("CameraStream.stop: Successfully acquired temp camera instance for reset. Releasing it.")
                    temp_camera.release()
                else:
                    print("CameraStream.stop: Failed to acquire temp camera instance for reset (might be good if already released).")
            except Exception as e:
                print(f"CameraStream.stop: Error during aggressive camera reset: {e}")
            # ----------------------------------------------

            if self.thread and self.thread.is_alive():
                print("CameraStream: Waiting for thread to finish (join timeout=5s)...")
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    print("CameraStream: Warning! Thread did not terminate gracefully within timeout. It might be stuck.")
                else:
                    print("CameraStream: Thread terminated successfully.")
            self.thread = None
            print("CameraStream thread stopped.")

# Instantiate the camera stream manager
camera_manager = CameraStream()

# Renamed get_camera to _get_camera_instance to emphasize it returns a new instance
def _get_camera_instance():
    with camera_lock: # Still use the lock to prevent multiple threads from initializing cameras concurrently
        print("Attempting to create and test a new camera instance (using GStreamer pipeline)...")
        
        # Re-introduce the specific GStreamer pipeline
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "appsink drop=true sync=false wait-on-eos=false"
        )


        try:
            print(f"Opening camera with GStreamer pipeline: {gst_pipeline}")
            cam_instance = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if cam_instance.isOpened():
                # No need to set properties explicitly, as they are in the GStreamer pipeline.
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
                    print(f"Camera instance successfully created and tested with GStreamer pipeline. Read {frames_read_successfully} test frames.")
                    return cam_instance
                else:
                    print("Not enough test frames could be read with GStreamer pipeline. Releasing new camera instance.")
                    cam_instance.release()
                    return None
            else:
                print("Camera instance not opened with GStreamer pipeline.")
                return None

        except Exception as e:
            print(f"Error during GStreamer camera instance creation: {e}")
            if 'cam_instance' in locals() and cam_instance.isOpened():
                cam_instance.release()
            return None

def release_camera():
    global camera_stream_running
    if camera_stream_running:
        print("release_camera: Stopping CameraStream thread. This will trigger internal camera release.")
        camera_manager.stop()
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
    try:
        # Get model instance
        model_instance = get_model()
        if model_instance is None:
            print("detect_defects: Failed to get model instance")
            with latest_detection_info_lock:
                latest_detection_info['defect_info'] = 'Model not available'
                latest_detection_info['defect_types'] = []
                latest_detection_info['confidences'] = []
            return frame

        # Run inference and visualization
        result_image, boxes, scores, class_ids = model_instance.infer_and_visualize(
            frame,
            conf_threshold=0.3,
            nms_threshold=0.4,
            save_path=None
        )

        # Update the defect info in the frame and global variable
        if boxes:
            defect_types = [model_instance.class_names.get(c, str(c)) for c in class_ids]
            confidences = [f'{s*100:.2f}%' for s in scores]
            defect_info = f"Detected: {', '.join(defect_types)} ({', '.join(confidences)})"
            # Log each detection in the history
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            for dt, conf in zip(defect_types, confidences):
                latest_detection_history.append({'type': dt, 'confidence': conf, 'timestamp': timestamp})
        else:
            defect_types = []
            confidences = []
            defect_info = "No defects detected"

        # Add text overlay to the frame
        cv2.putText(
            result_image,
            defect_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Update global detection info
        with latest_detection_info_lock:
            latest_detection_info['defect_info'] = defect_info
            latest_detection_info['defect_types'] = defect_types
            latest_detection_info['confidences'] = confidences

        return result_image

    except Exception as e:
        print(f"Error in detect_defects: {e}")
        with latest_detection_info_lock:
            latest_detection_info['defect_info'] = f'Error: {e}'
            latest_detection_info['defect_types'] = []
            latest_detection_info['confidences'] = []
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
            'result_image': result_filename,
            # original_image_path will be set by the upload_page route
        }
        return summary

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def generate_frames():
    global latest_frame
    print("generate_frames: Starting frame generation loop (real-time analysis)...")
    while True:
        try:
            if not camera_manager.running:
                print("generate_frames: CameraStream is not running. Attempting to start.")
                camera_manager.start()
                time.sleep(1)
            if latest_frame is None:
                print("generate_frames: No frames in buffer yet. Waiting...")
                time.sleep(0.5)
                continue
            with latest_frame_lock:
                frame_to_process = latest_frame.copy()
            # Perform real-time inference and draw bounding boxes
            processed_frame = detect_defects(frame_to_process)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                print("generate_frames: Failed to encode frame.")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
        except Exception as e:
            print(f"generate_frames: Error in loop: {e}")
            time.sleep(1)
            continue

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
        # Stop camera stream FIRST, then release model
        release_camera() # This will now aggressively stop the CameraStream thread
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
                
                # Generate a unique filename for the original image
                original_filename = f"original_{int(time.time())}_{file.filename}"
                original_save_path = os.path.join('static/uploaded_originals', original_filename)
                file.save(original_save_path)
                print(f"Original file saved to {original_save_path}")
                
                # Process the image from the saved original path
                result = process_image(original_save_path)
                print(f"Processing result: {result}")
                
                # No longer delete the original file here as it's saved in static/uploaded_originals for display
                
                if result is None:
                    return render_template('upload.html', result=None, error="Error processing image")
                
                # Pass both the detection results and the original image URL to the template
                return render_template('upload.html', result=result, original_image_url=url_for('static', filename='uploaded_originals/' + original_filename))
                
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

#ABOVEMAIN
@app.route('/live_status')
def live_status():
    # Return a shallow copy of the info and history without locking
    info = dict(latest_detection_info)
    info['history'] = list(latest_detection_history)
    print('LIVE STATUS RESPONSE:', info)
    return jsonify(info)

if __name__ == '__main__':
    try:
        print("Application starting...")
        release_camera() # Ensure camera manager is stopped at startup
        release_model()
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) # threaded=True to allow concurrent requests
    finally:
        print("Application exiting. Ensuring resources are released.")
        release_camera()
        release_model() 