import cv2
print("OpenCV imported successfully")
print("Version:", cv2.__version__)

# Try to create a simple image
img = cv2.imread('/dev/video0')
if img is None:
    print("Could not read image")
else:
    print("Successfully read image") 