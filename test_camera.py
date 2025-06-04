#!/usr/bin/env python3
import cv2
import time
import os

def test_camera():
    print("Testing camera...")
    
    # Check if we can access the camera device
    if not os.path.exists('/dev/video0'):
        print("Error: Camera device not found")
        return
        
    print("Camera device found")
    
    try:
        # Try to open camera with V4L2 backend
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("Camera opened successfully")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print("Successfully read a frame")
            # Save the frame to verify it works
            cv2.imwrite('test_frame.jpg', frame)
            print("Saved test frame to test_frame.jpg")
        else:
            print("Error: Could not read frame")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
            print("Camera released")

if __name__ == "__main__":
    test_camera() 