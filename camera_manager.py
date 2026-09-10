"""
Mode B — Live Webcam Manager & WebSocket Broadcaster
Runs continuous camera capture, invokes YOLO detection engine, serves MJPEG stream,
and broadcasts WebSocket detection events and violence score ticks to connected frontend clients.
"""

import cv2
import time
import asyncio
import numpy as np
from typing import Set, Optional
from fastapi import WebSocket
try:
    from backend.detector import RoadDetectionEngine, VideoState
    from backend.database import Database
    from backend.schemas import DetectionEvent, ViolenceScoreTick
except ImportError:
    from detector import RoadDetectionEngine, VideoState
    from database import Database
    from schemas import DetectionEvent, ViolenceScoreTick


class WebcamManager:
    """
    Single global live webcam manager for Mode B.
    Captures live frames, runs YOLO inference, maintains current JPEG frame,
    and broadcasts events over WebSocket connections.
    """

    def __init__(self, engine: RoadDetectionEngine, db: Database, camera_index: int = 0):
        self.engine = engine
        self.db = db
        self.camera_index = camera_index

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.latest_jpeg: Optional[bytes] = None

        self.active_websockets: Set[WebSocket] = set()
        self.state = VideoState(required_frames=self.engine.settings.requiredFrames)
        self.frame_counter = 0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[WebcamManager] Warning: Could not open camera index {self.camera_index}. Live webcam will remain idle until connected.")
        else:
            print(f"[WebcamManager] Camera index {self.camera_index} opened successfully.")

    def stop(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        print("[WebcamManager] Camera stream stopped.")

    async def register_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.add(websocket)
        print(f"[WebcamManager] Client connected to WS. Total active: {len(self.active_websockets)}")

    def unregister_websocket(self, websocket: WebSocket):
        self.active_websockets.discard(websocket)
        print(f"[WebcamManager] Client disconnected from WS. Remaining active: {len(self.active_websockets)}")

    async def broadcast_json(self, data: dict):
        if not self.active_websockets:
            return
        disconnected = set()
        for ws in self.active_websockets:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.unregister_websocket(ws)

    async def capture_loop(self):
        """Async background loop grabbing frames, running YOLO, and broadcasting ticks."""
        print("[WebcamManager] Starting async frame capture loop...")
        self.start()

        while True:
            if not self.is_running:
                await asyncio.sleep(0.5)
                continue

            if not self.cap or not self.cap.isOpened():
                # Attempt to re-open camera periodically
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    await asyncio.sleep(2.0)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            self.frame_counter += 1
            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

            # Run detection engine
            annotated_frame, events, tick = self.engine.process_frame(
                frame=frame,
                state=self.state,
                frame_id=self.frame_counter,
                timestamp_iso=ts_iso,
                save_snapshot=True,
            )

            # Encode frame to JPEG for MJPEG stream
            ret_encode, jpeg_buf = cv2.imencode(".jpg", annotated_frame)
            if ret_encode:
                self.latest_jpeg = jpeg_buf.tobytes()

            # Save events to database and broadcast over WebSockets
            for ev in events:
                self.db.add_event(ev)
                await self.broadcast_json(ev.dict(by_alias=True))

            # Broadcast violence tick over WebSockets
            await self.broadcast_json(tick.dict())

            # ~30 fps tick rate
            await asyncio.sleep(0.03)

    def get_mjpeg_frame(self) -> Optional[bytes]:
        return self.latest_jpeg
