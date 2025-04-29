import cv2
import numpy as np
import base64
import json
import requests
import psutil
import time
import asyncio
import websockets
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import URLS, PATHS, SYSTEM, QUALITY_MAP
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_device.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("edge_device")

logger.info(f"Loading YOLO model from {PATHS['yolo_model_edge']}")
model = YOLO(PATHS["yolo_model_edge"])

# initial values
target_fps = SYSTEM["default_target_fps"]
frame_skip = SYSTEM["default_frame_skip"]
roi_width = SYSTEM["default_roi_width"] 
roi_height = SYSTEM["default_roi_height"]
cpu_threshold = SYSTEM["default_cpu_threshold"]
processing_mode = SYSTEM["default_processing_mode"]
video_quality = SYSTEM["default_video_quality"]

logger.info(f"Initialized settings: FPS={target_fps}, Frame Skip={frame_skip}, ROI={roi_width}x{roi_height}, "
            f"CPU Threshold={cpu_threshold}%, Mode={processing_mode}, Quality={video_quality}")

# FPS calculation
prev_time = time.time()

motion_enabled = SYSTEM["motion_detection"]["enabled"]
motion_threshold = SYSTEM["motion_detection"]["threshold"]
min_contour_area = SYSTEM["motion_detection"]["min_contour_area"]
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=SYSTEM["motion_detection"]["history"], 
    varThreshold=16, 
    detectShadows=False
)

def get_fps():
    global prev_time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 0.01)
    prev_time = curr_time
    return min(fps, target_fps)

# Fetch settings from the web interface
def fetch_settings():
    global target_fps, frame_skip, roi_width, roi_height, cpu_threshold, processing_mode, video_quality
    try:
        logger.info(f"Fetching settings from {URLS['settings_url']}")
        response = requests.get(URLS["settings_url"], timeout=1)
        if response.status_code == 200:
            settings = response.json()
            
            # Log each setting value before and after update
            logger.info(f"Current settings before update:")
            logger.info(f"  FPS Limit: {target_fps}")
            logger.info(f"  Frame Skip: {frame_skip}")
            logger.info(f"  ROI Width: {roi_width}")
            logger.info(f"  ROI Height: {roi_height}")
            logger.info(f"  CPU Threshold: {cpu_threshold}")
            logger.info(f"  Processing Mode: {processing_mode}")
            logger.info(f"  Video Quality: {video_quality}")
            
            # Update settings with received values
            new_fps = int(settings.get("fps_limit", target_fps))
            new_frame_skip = int(settings.get("frame_skip", frame_skip))
            new_roi_width = int(settings.get("roi_width", roi_width))
            new_roi_height = int(settings.get("roi_height", roi_height))
            new_cpu_threshold = int(settings.get("cpu_threshold", cpu_threshold))
            new_processing_mode = settings.get("processing_mode", processing_mode)
            new_video_quality = settings.get("video_quality", video_quality)
            
            if new_fps != target_fps:
                logger.info(f"Updating FPS limit: {target_fps} -> {new_fps}")
                target_fps = new_fps
            
            if new_frame_skip != frame_skip:
                logger.info(f"Updating Frame Skip: {frame_skip} -> {new_frame_skip}")
                frame_skip = new_frame_skip
            
            if new_roi_width != roi_width:
                logger.info(f"Updating ROI Width: {roi_width} -> {new_roi_width}")
                roi_width = new_roi_width
            
            if new_roi_height != roi_height:
                logger.info(f"Updating ROI Height: {roi_height} -> {new_roi_height}")
                roi_height = new_roi_height
            
            if new_cpu_threshold != cpu_threshold:
                logger.info(f"Updating CPU Threshold: {cpu_threshold} -> {new_cpu_threshold}")
                cpu_threshold = new_cpu_threshold
            
            if new_processing_mode != processing_mode:
                logger.info(f"Updating Processing Mode: {processing_mode} -> {new_processing_mode}")
                processing_mode = new_processing_mode
            
            if new_video_quality != video_quality:
                logger.info(f"Updating Video Quality: {video_quality} -> {new_video_quality}")
                video_quality = new_video_quality
            
            # Log final settings after update
            logger.info(f"Settings updated successfully")
        else:
            logger.error(f"Failed to fetch settings: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching settings: {e}")

# === CMAB: Contextual Multi-Armed Bandit ===
class CMAB:
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.q_values = {a: 0 for a in actions}
        self.action_counts = {a: 1 for a in actions}
        self.alpha = alpha
        self.gamma = gamma
        self.history = []
        logger.info(f"CMAB initialized with actions: {actions}")

    def select_action(self):
        """Select an action using ε-greedy strategy"""
        epsilon = 0.3
        if random.random() < epsilon:
            action = random.choice(self.actions)
            logger.debug(f"CMAB exploration: selected random action {action}")
            return action
        action = max(self.q_values, key=self.q_values.get)
        logger.debug(f"CMAB exploitation: selected best action {action}")
        return action

    def update(self, action, reward):
        """Update reward values for the chosen action"""
        old_value = self.q_values[action]
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        self.history.append(self.q_values.copy())
        logger.debug(f"CMAB updated {action}: {old_value:.2f} -> {self.q_values[action]:.2f} (reward: {reward:.2f})")

    def get_q_values(self):
        return self.q_values
    
    def get_history(self):
        return self.history

def detect_motion(frame, prev_frame=None):
    """
    Detect motion in the frame using background subtraction
    Returns: (motion_detected, fg_mask)
    """
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Threshold the mask
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour has significant area
    significant_motion = False
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            significant_motion = True
            break
    
    if significant_motion:
        logger.debug("Motion detected in frame")
    
    return significant_motion, fg_mask

# ROI processing
def apply_roi_mask(frame, roi_corners):
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_corners, dtype=np.int32)], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

async def send_frame_to_server(image):
    quality = QUALITY_MAP.get(video_quality, 50)
    compressed_image = base64.b64encode(cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]).decode("utf-8")
    try:
        logger.debug(f"Connecting to server at {URLS['server_ws']}")
        async with websockets.connect(URLS["server_ws"]) as ws:
            await ws.send(json.dumps({"image": compressed_image}))
            response = await ws.recv()
            return json.loads(response)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        return {"error": str(e)}

# Update web interface
def update_dashboard(fps_val, cpu_val, objects_val, mode_val, resolution="medium", q_values=None):
    try:
        payload = {
            "fps": fps_val,
            "cpu": cpu_val,
            "objects": objects_val,
            "mode": mode_val,
            "resolution": resolution
        }
        if q_values:
            payload["cmab_q_values"] = q_values
        logger.debug(f"Updating dashboard with FPS={fps_val}, CPU={cpu_val}%, Objects={objects_val}, Mode={mode_val}")
        requests.post(URLS["web_interface"], json=payload)
    except Exception as e:
        logger.error(f"Failed to update dashboard: {e}")

# Send processed frame to web interface
def send_frame_to_web(frame):
    quality = QUALITY_MAP.get(video_quality, 50)  # Default to medium
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    frame_bytes = base64.b64encode(buffer).decode('utf-8')
    try:
        requests.post(URLS["frame_update_url"], json={"frame": frame_bytes})
        logger.debug("Frame sent to web interface")
    except Exception as e:
        logger.error(f"Failed to send frame: {e}")


def main():
    logger.info("Starting edge device processing")
    cmab = CMAB(actions=["low_res", "medium_res", "high_res"])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Unable to open camera")
        return

    frame_count = 0
    prev_action = "medium_res"
    last_settings_check = time.time()
    prev_frame = None
    last_processed_time = time.time()
    motion_count = 0  # Counter for consecutive motion frames
    roi_update_applied = False
    
    logger.info("Camera initialized, entering main loop")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            # Periodically fetch settings (every 5 seconds)
            current_time = time.time()
            if current_time - last_settings_check >= 5:
                fetch_settings()
                last_settings_check = current_time
                roi_update_applied = False  # Reset flag to check if ROI is applied

            frame_count += 1
            
            # Safe check for frame_skip to avoid division by zero
            if frame_skip > 0 and frame_count % frame_skip != 0:
                continue

            # Check for motion if enabled
            if motion_enabled:
                motion_detected, fg_mask = detect_motion(frame)
                
                # Process frame only if motion detected or after every 15 frames
                if motion_detected:
                    motion_count += 1
                    logger.debug(f"Motion detected ({motion_count} consecutive frames)")
                else:
                    motion_count = 0
                    
                # Skip processing if no motion detected and we've processed recently
                # But still process at least one frame every 2 seconds for monitoring
                if not motion_detected and motion_count < 3 and current_time - last_processed_time < 2.0:
                    # Still send the frame to web without processing
                    send_frame_to_web(frame)
                    continue
                
                logger.debug("Processing frame (motion detected or monitoring frame)")
                last_processed_time = current_time
            
            # Select resolution using CMAB
            action = cmab.select_action()
            if action == "low_res":
                frame = cv2.resize(frame, (320, 240))
                resolution = "low"
                logger.debug("Using low resolution (320x240)")
            elif action == "medium_res":
                frame = cv2.resize(frame, (640, 480))
                resolution = "medium"
                logger.debug("Using medium resolution (640x480)")
            elif action == "high_res":
                frame = cv2.resize(frame, (1280, 720))
                resolution = "high"
                logger.debug("Using high resolution (1280x720)")

            # Define ROI based on updated settings
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            roi_corners = [
                (center_x - roi_width // 2, center_y - roi_height // 2),
                (center_x + roi_width // 2, center_y - roi_height // 2),
                (center_x + roi_width // 2, center_y + roi_height // 2),
                (center_x - roi_width // 2, center_y + roi_height // 2),
            ]
            
            # Log ROI information for every frame after settings update
            if not roi_update_applied:
                logger.info(f"Applying ROI: {roi_width}x{roi_height} centered at ({center_x}, {center_y})")
                logger.info(f"ROI corners: {roi_corners}")
                roi_update_applied = True
            
            frame_roi = apply_roi_mask(frame, roi_corners)

            results = model(frame_roi)
            detections = results[0].boxes
            num_objects = len(detections)
            
            logger.debug(f"Detected {num_objects} objects in ROI")

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += center_x - roi_width // 2
                y1 += center_y - roi_height // 2
                x2 += center_x - roi_width // 2
                y2 += center_y - roi_height // 2
                if (x1 >= center_x - roi_width // 2 and x2 <= center_x + roi_width // 2 and
                    y1 >= center_y - roi_height // 2 and y2 <= center_y + roi_height // 2):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw motion areas if motion detection is enabled
            if motion_enabled and 'fg_mask' in locals():
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > min_contour_area:
                        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)

            # TAODM: Decide processing location
            if processing_mode == "auto":
                threshold = cpu_threshold if num_objects < SYSTEM["object_threshold"] else SYSTEM["object_threshold"]
            elif processing_mode == "local":
                threshold = float('inf')  # Force local processing
            else:  # "server"
                threshold = 0  # Force server processing

            if num_objects >= threshold:
                mode = "server"
                logger.info(f"Sending frame to server - {num_objects} objects detected")
                response = asyncio.run(send_frame_to_server(frame))
                logger.info(f"Server response: {response}")
                num_objects = response.get("detections", num_objects)
            else:
                mode = "local"
                logger.debug(f"Processing locally - {num_objects} objects detected")

            # Update CMAB with reward (FPS) based on the previous action
            fps_val = get_fps()
            cmab.update(prev_action, fps_val)
            prev_action = action

            # Update dashboard with current stats and Q-values
            cpu_val = psutil.cpu_percent()
            q_values = cmab.get_q_values()
            update_dashboard(fps_val, cpu_val, num_objects, mode, resolution, q_values)

            # Visualize ROI
            cv2.polylines(frame, [np.array(roi_corners, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

            # Display current settings on frame
            settings_text = f"FPS: {int(fps_val)} | ROI: {roi_width}x{roi_height} | Mode: {mode}"
            cv2.putText(frame, settings_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Send the processed frame to the web interface
            send_frame_to_web(frame)

            # If motion is detected, add indicator in the corner
            if motion_enabled and motion_detected:
                cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)

            cv2.imshow("Device Processing", frame)

            prev_frame = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit key pressed, exiting loop")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Closing video...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")
        
        # Display CMAB learning progress
        history = cmab.get_history()
        if history:
            logger.info("Generating CMAB learning graph")
            plt.figure(figsize=(10, 5))
            for act in cmab.actions:
                plt.plot([h[act] for h in history], label=act)
            plt.xlabel("Iteration")
            plt.ylabel("Q-Value")
            plt.legend()
            plt.title("Evolution of Q-Values in CMAB")
            plt.savefig(PATHS["cmab_learning_plot"])
            logger.info(f"CMAB learning progress saved as '{PATHS['cmab_learning_plot']}'")
        else:
            logger.warning("No CMAB history to plot")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)