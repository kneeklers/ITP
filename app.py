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

# Load the model once at startup for efficiency
MODEL_PATH = 'models/your_model.engine'

# Create uploads and results directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            try:
                # Try different camera backends
                for backend in [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]:
                    camera = cv2.VideoCapture(0, backend)
                    if camera.isOpened():
                        # Set camera properties
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        # Try to read a test frame
                        ret, _ = camera.read()
                        if ret:
                            print(f"Camera initialized successfully with backend {backend}")
                            return camera
                        else:
                            camera.release()
                            camera = None
            except Exception as e:
                print(f"Error initializing camera: {e}")
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
    # Placeholder for live defect detection
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
    while True:
        try:
            with camera_lock:
                camera = get_camera()
                if camera is None:
                    print("Camera not available")
                    break
                
                success, frame = camera.read()
                if not success:
                    print("Failed to read frame")
                    # Try to reinitialize camera
                    release_camera()
                    camera = get_camera()
                    if camera is None:
                        break
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"Streaming frame {frame_count}")
                
                # Process frame if needed
                processed_frame = detect_defects(frame)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    print("Failed to encode frame")
                    continue
                
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break
    
    # Clean up if loop breaks
    release_camera()

@app.route('/')
def index():
    try:
        # Release model when switching to live stream
        release_model()
        
        # Initialize camera
        camera = get_camera()
        if camera is None:
            return "Error: Could not initialize camera", 500
        
        return render_template('index.html')
    except Exception as e:
        print(f"Error in index route: {e}")
        return "Error initializing camera", 500

@app.route('/video_feed')
def video_feed():
    try:
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