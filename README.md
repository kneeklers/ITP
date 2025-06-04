# Metal Sheet Surface Defect Detection System

This project implements a real-time metal sheet surface defect detection system using a Jetson Nano and a web interface for remote monitoring.

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/         # HTML templates
│   └── index.html     # Main web interface
└── README.md          # This file
```

## Setup Instructions

### On Jetson Nano:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure your camera is properly connected to the Jetson Nano.

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. The server will start on `http://0.0.0.0:5000`

### On Client Computer:

1. Open a web browser and navigate to:
   ```
   http://<jetson-nano-ip-address>:5000
   ```
   Replace `<jetson-nano-ip-address>` with the actual IP address of your Jetson Nano.

## Features

- Real-time video streaming from the Jetson Nano camera
- Live defect detection (to be implemented in the `detect_defects` function)
- Web interface accessible from any device on the network
- Responsive design using Bootstrap

## Implementation Notes

1. The `detect_defects` function in `app.py` is currently a placeholder. You'll need to implement your defect detection algorithm there.

2. The camera index in `app.py` is set to 0 by default. If you're using a different camera, modify the `cv2.VideoCapture(0)` line accordingly.

3. The web interface is designed to be responsive and will work on both desktop and mobile devices.

## Security Considerations

- The current implementation runs on all network interfaces (`0.0.0.0`). For production use, consider implementing proper authentication and security measures.
- Make sure your network is secure when accessing the stream remotely.

## Troubleshooting

1. If the video feed doesn't appear:
   - Check if the camera is properly connected
   - Verify the camera index in `app.py`
   - Ensure no other application is using the camera

2. If you can't access the web interface:
   - Verify the Jetson Nano's IP address
   - Check if port 5000 is not blocked by a firewall
   - Ensure both devices are on the same network 