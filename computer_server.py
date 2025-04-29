import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO
from config import NETWORK, PATHS


model = YOLO(PATHS["yolo_model_server"])

async def process_image(websocket):
    client_ip = websocket.remote_address[0]
    print(f"Client connected: {client_ip}")
    async for message in websocket:
        try:
            data = json.loads(message)
            if "image" not in data:
                await websocket.send(json.dumps({"error": "Missing image data"}))
                continue
            image_data = base64.b64decode(data["image"])
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = model(frame)
            detections = len(results[0].boxes)

            response = json.dumps({"detections": detections})
            await websocket.send(response)
        except Exception as e:
            print(f"Error processing frame from {client_ip}: {e}")

async def main():
    server_port = NETWORK["server_port"]
    server_ip = "0.0.0.0"  # Listen on all interfaces
    async with websockets.serve(process_image, server_ip, server_port):
        print(f"WebSocket server running on ws://{NETWORK['server_ip']}:{server_port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())