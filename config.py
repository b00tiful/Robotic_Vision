
NETWORK = {
    "server_ip": "127.0.0.1",
    "server_port": 5555,
    "web_ip": "127.0.0.1",
    "web_port": 5556,
}

URLS = {
    "server_ws": f"ws://{NETWORK['server_ip']}:{NETWORK['server_port']}",
    "web_interface": f"http://{NETWORK['web_ip']}:{NETWORK['web_port']}/update",
    "settings_url": f"http://{NETWORK['web_ip']}:{NETWORK['web_port']}/settings",
    "frame_update_url": f"http://{NETWORK['web_ip']}:{NETWORK['web_port']}/update_frame",
}

PATHS = {
    "yolo_model_edge": "yolov8n.pt",
    "yolo_model_server": "yolov8m.pt",
    "templates_dir": "templates",
    "cmab_learning_plot": "cmab_learning.png",
}

SYSTEM = {
    "object_threshold": 2,
    "default_target_fps": 10,
    "default_frame_skip": 2,
    "default_roi_width": 640,
    "default_roi_height": 480,
    "default_cpu_threshold": 85,
    "default_processing_mode": "auto",
    "default_video_quality": "medium",
    "motion_detection": {
        "enabled": True,
        "threshold": 1000,
        "min_contour_area": 500,
        "history": 3,
    }
}

QUALITY_MAP = {
    "low": 30,
    "medium": 50,
    "high": 80
}

DEFAULT_SETTINGS = {
    "fps_limit": 15,
    "frame_skip": 1,
    "roi_width": 640,
    "roi_height": 480,
    "auto_mode_switch": "enabled",
    "cpu_threshold": 85,
    "processing_mode": "auto",
    "video_quality": "medium"
}