# Robotic_Vision
Dynamic Real-Time Video Processing System

A system for real-time video processing with edge and server components, utilizing YOLO for object detection, WebSocket for communication, and a Flask-based web interface for monitoring and control. Features dynamic resolution adjustment via a Contextual Multi-Armed Bandit (CMAB) algorithm, motion detection, and adaptive processing mode switching.

Requirements

    Python 3.8+
    Libraries: opencv-python, ultralytics, websockets, flask, requests, psutil, numpy, matplotlib
    YOLOv8 models (yolov8n.pt, yolov8m.pt)
    Web browser for accessing the interface

Setup

    1. Clone repository
        git clone <repository-url>
        cd <repository-directory>

    2. Install dependencies
        pip install -r requirements.txt

    3. Download YOLO models
        Ensure yolov8n.pt and yolov8m.pt are available in the project directory or update PATHS in config.py

    4. Configure network
        Update NETWORK in config.py with appropriate IP addresses and ports for your setup

    5. Run the system
        Start the Websocket server
            python computer_server.py
        Start the Web interface
            python web_interface.py
        Run the main code on edge device
            python full_model.py
            
    6. Access the Web interface
        Open a browser and navigate to http://<web_ip>:<web_port>

Usage

    Monitor: View real-time stats, charts, and video feed via the web interface.
    Adjust settings: Modify FPS, ROI, processing mode, etc., through the settings modals.
    Export data: Save statistics as JSON or charts as PNG.
    View logs: Check system logs for debugging.
    Keyboard shortcuts:
        Alt + 1-5: Open stats, charts, video, settings or logs.
        Alt + D: Toggle light/dark theme.
        Esc: Close modals.

Note

    Ensure the camera is accessible for full_model.py.
    Adjust SYSTEM parameters in config.py for performance tuning.
    Logs are stored in edge_device.log and web_interface.log.
    CMAB learning progress is saved as cmab_learning.png.