from flask import Flask, render_template, jsonify, request, Response
import os
import time
import cv2
import base64
from config import NETWORK, DEFAULT_SETTINGS, PATHS
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_interface")

app = Flask(__name__)

fps = 0
cpu_load = 0
objects_detected = 0
processing_mode = "local"
frame_size = "640x480"
roi_size = "320x240"
frame_skip = 0
uptime = 0
logs = []
latest_frame = None
cmab_q_values = {"low_res": 0, "medium_res": 0, "high_res": 0}

settings = DEFAULT_SETTINGS.copy()
logger.info(f"Initial settings loaded: {settings}")

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    logs.append(log_entry)
    if len(logs) > 100:
        logs.pop(0)
    logger.info(message)

@app.route('/')
def index():
    logger.info("Main page accessed")
    return render_template('index.html')

@app.route('/stats')
def get_stats():
    return jsonify({
        "fps": fps,
        "cpu": cpu_load,
        "objects": objects_detected,
        "mode": processing_mode,
        "frame_size": frame_size,
        "roi_size": roi_size,
        "frame_skip": frame_skip,
        "uptime": uptime,
        "cmab_q_values": cmab_q_values
    })

@app.route('/update', methods=['POST'])
def update_stats():
    global fps, cpu_load, objects_detected, processing_mode, uptime, cmab_q_values
    data = request.json
    fps = data.get("fps", fps)
    cpu_load = data.get("cpu", cpu_load)
    objects_detected = data.get("objects", objects_detected)
    processing_mode = data.get("mode", processing_mode)
    frame_size = data.get("resolution", "640x480")
    if "cmab_q_values" in data:
        cmab_q_values = data["cmab_q_values"]
    uptime += 1
    add_log(f"Stats updated - FPS: {fps}, CPU: {cpu_load}%, Objects: {objects_detected}, Mode: {processing_mode}, CMAB: {cmab_q_values}")
    return jsonify({"status": "updated"})

@app.route('/update_frame', methods=['POST'])
def update_frame():
    global latest_frame
    data = request.json
    latest_frame = data.get("frame")
    logger.debug("Received frame update")
    return jsonify({"status": "frame updated"})

@app.route('/settings', methods=['GET', 'POST'])
def handle_settings():
    global settings, roi_size, frame_skip
    if request.method == 'POST':
        data = request.json
        logger.info(f"Settings update requested with data: {data}")
        logger.info(f"Current settings before update: {settings}")
        
        for key, new_value in data.items():
            if key in settings:
                old_value = settings[key]
                if new_value != old_value:
                    logger.info(f"Setting change - {key}: {old_value} -> {new_value}")
        
        settings.update(data)
        roi_size = f"{settings['roi_width']}x{settings['roi_height']}"
        frame_skip = settings["frame_skip"]
        
        logger.info(f"Settings after update: {settings}")
        add_log(f"Settings updated: {data}")
        return jsonify({"status": "settings updated", "settings": settings})
    
    logger.info(f"Settings requested by client: {settings}")
    add_log("Settings fetched by client")
    return jsonify(settings)

@app.route('/logs')
def get_logs():
    return jsonify({"logs": logs})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            if latest_frame:
                frame_bytes = base64.b64decode(latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    logger.info("Video feed requested")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export_stats')
def export_stats():
    stats_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": fps,
        "cpu": cpu_load,
        "objects": objects_detected,
        "mode": processing_mode,
        "frame_size": frame_size,
        "roi_size": roi_size,
        "frame_skip": frame_skip,
        "uptime": uptime
    }
    logger.info(f"Stats export requested: {stats_data}")
    return jsonify(stats_data)

if __name__ == "__main__":
    templates_dir = PATHS["templates_dir"]
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f"Created templates directory: {templates_dir}")
    
    host = NETWORK["web_ip"]
    port = NETWORK["web_port"]
    logger.info(f"Starting web interface server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)