import cv2
import numpy as np
import json
import base64
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

class ArUcoDetector:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 11
        self.parameters.adaptiveThreshWinSizeStep = 2
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = 0.06
        self.parameters.maxMarkerPerimeterRate = 1.5
        self.parameters.polygonalApproxAccuracyRate = 0.06
        self.parameters.minCornerDistanceRate = 0.06
        self.parameters.minDistanceToBorder = 1
        self.parameters.minMarkerDistanceRate = 0.06
        print("Ultra-fast ArUco detector initialized with FastAPI WebSocket support")

    def detect_markers_from_base64(self, base64_data):
        try:
            start_time = time.time()
            img_data = base64.b64decode(base64_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {'error': 'Failed to decode image', 'markers': []}
            height, width = image.shape
            if width > 400:
                scale = 400 / width
                new_width = 400
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                scale_factor = 1 / scale
            else:
                scale_factor = 1
                new_width, new_height = width, height
            image = cv2.GaussianBlur(image, (3, 3), 0)
            detection_start = time.time()
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, rejected = detector.detectMarkers(image)
            detection_time = time.time() - detection_start
            markers = []
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    corner_points = corners[i][0] * scale_factor
                    center_x = np.mean(corner_points[:, 0])
                    center_y = np.mean(corner_points[:, 1])
                    marker_data = {
                        'id': int(marker_id),
                        'corners': corner_points.tolist(),
                        'center': {'x': float(center_x), 'y': float(center_y)}
                    }
                    markers.append(marker_data)
            total_time = time.time() - start_time
            return {
                'markers': markers,
                'detection_time': round(detection_time * 1000, 1),
                'total_time': round(total_time * 1000, 1),
                'processed_size': f"{new_width}x{new_height}",
                'total_markers': len(markers),
                'scale': scale_factor,
            }
        except Exception as e:
            return {
                'error': str(e),
                'markers': [],
                'detection_time': 0,
                'total_time': 0
            }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ArUcoDetector()
stats = {
    'total_frames': 0,
    'total_markers': 0,
    'start_time': time.time()
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({
        'type': 'connection',
        'status': 'connected',
        'message': 'ArUco FastAPI WebSocket server ready'
    }))
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                if data.get('type') == 'frame':
                    frame_start = time.time()
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, detector.detect_markers_from_base64, data['frame']
                    )
                    result['websocket_latency'] = round((time.time() - frame_start) * 1000, 1)
                    result['timestamp'] = data.get('timestamp', time.time() * 1000)
                    result['type'] = 'detection_result'
                    stats['total_frames'] += 1
                    stats['total_markers'] += result.get('total_markers', 0)
                    await websocket.send_text(json.dumps(result))
                elif data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': data.get('timestamp', time.time() * 1000)
                    }))
                elif data.get('type') == 'stats':
                    uptime = time.time() - stats['start_time']
                    await websocket.send_text(json.dumps({
                        'type': 'server_stats',
                        'total_frames': stats['total_frames'],
                        'total_markers': stats['total_markers'],
                        'uptime': round(uptime, 1),
                        'fps': round(stats['total_frames'] / uptime, 1) if uptime > 0 else 0,
                    }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'Processing error: {str(e)}'
                }))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("temp:app", host="0.0.0.0", port=8001, reload=True)