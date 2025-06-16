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
model = None
model_lock = threading.Lock()

# GStreamer pipeline for camera (IMPORTANT: Choose one based on your camera type)
# For CSI camera (e.g., Raspberry Pi Camera Module V2):
# GSTREAMER_PIPELINE = "nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

# For USB camera (replace /dev/video0 with your camera device if different):
# GSTREAMER_PIPELINE = "v4l2src device=/dev/video0 ! video/x-raw, width=(int)640, height=(int)480, framerate=(fraction)30/1 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

# Default to None, user MUST uncomment and set one
# GSTREAMER_PIPELINE = None # This line should now be commented out or removed
GSTREAMER_PIPELINE = "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false"

# Load the model once at startup for efficiency
MODEL_PATH = 'models/your_model.engine'

# Create uploads and results directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            print("Attempting to initialize camera...")
            if GSTREAMER_PIPELINE is None:
                print("ERROR: GSTREAMER_PIPELINE is not set. Please uncomment and configure it in app.py")
                return None

            try:
                print(f"Opening camera with GStreamer pipeline: {GSTREAMER_PIPELINE}")
                camera = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
                
                if camera.isOpened():
                    # No need to set properties like width, height, fps here, as they are defined in the GStreamer pipeline.
                    # We'll just confirm it's opened and try to read a few frames.
                    
                    # Try to read several test frames to ensure stream is active
                    test_frames_to_read = 5 # Reduced test frames for faster startup
                    frames_read_successfully = 0
                    for i in range(test_frames_to_read):
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            frames_read_successfully += 1
                            # print(f"Read test frame {i+1}/{test_frames_to_read} successfully. Frame shape: {frame.shape}") # Keep this if more verbose logging is needed
                            time.sleep(0.01) # Small delay to allow buffer to fill
                        else:
                            print(f"Failed to read test frame {i+1}/{test_frames_to_read} from GStreamer pipeline. Ret: {ret}, Frame is None: {frame is None}")
                            
                    if frames_read_successfully >= 3: # Require at least 3 successful frames to consider it stable
                        print(f"Camera initialized successfully using GStreamer pipeline. Read {frames_read_successfully} test frames.")
                        return camera
                    else:
                        print("Not enough test frames could be read with GStreamer pipeline. Releasing camera.")
                        camera.release()
                        camera = None
                else:
                    print("Camera not opened with GStreamer pipeline.")
                    camera = None 

            except Exception as e:
                print(f"Error during GStreamer camera initialization: {e}")
                if camera is not None:
                    camera.release()
                    camera = None
                return None
    return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                camera = None
            except Exception as e:
                print(f"Error releasing camera: {e}")

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
                # Clean up CUDA context
                try:
                    import pycuda.driver as cuda
                    cuda.Context.pop()
                except:
                    pass
                
                del model
                model = None
                print("Model released successfully")
            except Exception as e:
                print(f"Error releasing model: {e}")

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
    frame_count = 0
    consecutive_failures = 0
    max_failures = 5
    
    print("Starting frame generation loop...")
    
    while True:
        try:
            # Ensure camera is acquired within the lock to prevent race conditions
            with camera_lock:
                camera = get_camera()
                if camera is None:
                    print("generate_frames: Camera not available, retrying...")
                    time.sleep(1)  # Wait before retrying camera acquisition
                    consecutive_failures += 1 # Treat as a failure to get camera
                    if consecutive_failures >= max_failures:
                        print("generate_frames: Max consecutive camera failures reached. Exiting stream.")
                        break
                    continue
                
                # Attempt to read a frame
                ret, frame = camera.read()
                
                if not ret or frame is None:
                    print(f"generate_frames: Failed to read frame. Ret: {ret}, Frame is None: {frame is None}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("generate_frames: Too many consecutive frame read failures, reinitializing camera.")
                        release_camera()
                        consecutive_failures = 0
                        time.sleep(1)  # Wait before retrying camera reinitialization
                    continue
                
                consecutive_failures = 0  # Reset on successful frame read
                frame_count += 1
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"generate_frames: Streaming frame {frame_count}, size: {frame.shape}")
                
                # Process frame if needed (this should be fast)
                processed_frame = detect_defects(frame)
                
                # Encode frame with lower quality for better performance
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    print("generate_frames: Failed to encode frame.")
                    continue
                
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"generate_frames: Error in loop: {e}")
            time.sleep(1)  # Wait before retrying
            # Consider breaking if the error is persistent and unrecoverable
            # For now, let's keep retrying to see if it recovers
            continue
    
    # If the loop breaks, ensure camera is released
    print("generate_frames: Exiting loop. Releasing camera.")
    release_camera()

@app.route('/')
def index():
    try:
        print("Accessing index route")
        # Release model when switching to live stream
        release_model()
        
        # Ensure camera is initialized before rendering the page
        # This will block until the camera is ready or fails
        camera = get_camera()
        if camera is None:
            print("Failed to initialize camera in index route. Showing error page.")
            return "Error: Could not initialize camera. Check console for details.", 500
        
        print("Successfully initialized camera in index route")
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
        # Release camera and ensure model is released before switching to upload page
        release_camera()
        release_model()  # Ensure model is released before processing
        
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
                
                # Create uploads directory if it doesn't exist
                os.makedirs('uploads', exist_ok=True)
                
                # Save the file
                filename = os.path.join('uploads', file.filename)
                file.save(filename)
                print(f"File saved to {filename}")
                
                # Process the image
                result = process_image(filename)
                print(f"Processing result: {result}")
                
                # Clean up
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
        release_camera()
        release_model()
        return "Resources released"
    except Exception as e:
        print(f"Error in shutdown: {e}")
        return "Error releasing resources", 500

if __name__ == '__main__':
    try:
        # Ensure clean state at startup
        release_camera()
        release_model()
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
    finally:
        release_camera()
        release_model() 