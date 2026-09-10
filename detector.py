"""
Core YOLO & Violence Detection Engine
Refactored from detect.py and violence.py

Key improvements over original:
- Per-class cooldown deduplication: prevents emitting hundreds of events for
  the same pothole visible across consecutive frames.
- Best-box-per-class-per-frame: only the highest-confidence detection per class
  per frame is emitted, not every overlapping box.
"""

import os
import uuid
import time
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Any
try:
    from backend.schemas import (
        DetectionEvent,
        BoundingBox,
        PoseKeypoint,
        ViolenceScoreTick,
        DetectionClass,
        DetectionSettings,
    )
except ImportError:
    from schemas import (
        DetectionEvent,
        BoundingBox,
        PoseKeypoint,
        ViolenceScoreTick,
        DetectionClass,
        DetectionSettings,
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Per-class cooldown (in frames) — prevents repeated events for same object
DEFAULT_COOLDOWNS: Dict[str, int] = {
    "pothole": 30,   # ~1 second at 30fps
    "accident": 60,  # ~2 seconds
    "violence": 15,  # ~0.5 seconds (fast-changing)
}


class VideoState:
    """Per-session state tracking for temporal violence heuristic."""

    def __init__(self, required_frames: int = 6):
        self.previous_wrists: Dict[int, Tuple[float, float]] = {}
        self.suspicious_frames: int = 0
        self.required_frames: int = required_frames

        # Cooldown tracking: class_name -> last frame_id that emitted an event
        self.last_event_frame: Dict[str, int] = {}

    def reset(self):
        self.previous_wrists.clear()
        self.suspicious_frames = 0
        self.last_event_frame.clear()

    def can_emit(self, cls_name: str, frame_id: int) -> bool:
        """Check if enough frames have passed since last event for this class."""
        cooldown = DEFAULT_COOLDOWNS.get(cls_name, 30)
        last = self.last_event_frame.get(cls_name, -cooldown - 1)
        return (frame_id - last) >= cooldown

    def record_emission(self, cls_name: str, frame_id: int):
        """Record that an event was emitted for this class at this frame."""
        self.last_event_frame[cls_name] = frame_id


class RoadDetectionEngine:
    """Wraps YOLO pose tracking and road issue object detection."""

    def __init__(
        self,
        pose_model_path: str = "yolo11n-pose.pt",
        road_model_path: str = "Road Yolo.pt",
        snapshot_dir: str = "backend/static/snapshots",
        settings: Optional[DetectionSettings] = None,
    ):
        self.settings = settings or DetectionSettings()
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Device selection: CUDA vs CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RoadDetectionEngine] Initializing on device: {self.device.upper()}")

        # Load models
        print(f"[RoadDetectionEngine] Loading pose model from '{pose_model_path}'...")
        self.pose_model = YOLO(pose_model_path)
        print(f"[RoadDetectionEngine] Loading road model from '{road_model_path}'...")
        self.road_model = YOLO(road_model_path)
        print("[RoadDetectionEngine] Models loaded successfully.")

    def update_settings(self, new_settings: DetectionSettings):
        self.settings = new_settings

    def process_frame(
        self,
        frame: np.ndarray,
        state: VideoState,
        frame_id: int = 0,
        timestamp_iso: Optional[str] = None,
        save_snapshot: bool = True,
    ) -> Tuple[np.ndarray, List[DetectionEvent], ViolenceScoreTick]:
        """
        Processes a single video frame.
        Returns:
            - annotated_frame (np.ndarray)
            - list of DetectionEvent objects (deduplicated with cooldowns)
            - ViolenceScoreTick object
        """
        if timestamp_iso is None:
            timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

        # 1. Pose tracking
        pose_results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.settings.poseConfidence,
            verbose=False,
            device=self.device,
        )

        # 2. Road issues detection
        road_results = self.road_model.predict(
            source=frame,
            conf=self.settings.detectionConfidence,
            verbose=False,
            device=self.device,
        )

        pose_res = pose_results[0]
        road_res = road_results[0]

        violent_frame = False
        pose_keypoints_all: List[List[PoseKeypoint]] = []

        # 3. Violence Heuristic Logic (exact port from violence.py)
        if pose_res.keypoints is not None and pose_res.boxes is not None and pose_res.boxes.id is not None:
            try:
                keypoints = pose_res.keypoints.xy.cpu().numpy()
                ids = pose_res.boxes.id.cpu().numpy().astype(int)

                # Record keypoints for frontend display if needed
                for person_kpts in keypoints:
                    person_list = [
                        PoseKeypoint(x=float(pt[0]), y=float(pt[1]), confidence=1.0)
                        for pt in person_kpts
                    ]
                    pose_keypoints_all.append(person_list)

                if len(keypoints) >= 2:
                    for i in range(len(keypoints)):
                        for j in range(len(keypoints)):
                            if i == j:
                                continue

                            person_a = keypoints[i]
                            person_b = keypoints[j]
                            id_a = ids[i]

                            nose_a = person_a[0]
                            nose_b = person_b[0]
                            left_wrist = person_a[9]
                            right_wrist = person_a[10]

                            # 1. Person distance check
                            person_dist = distance(nose_a, nose_b)
                            if person_dist > self.settings.personDistanceThreshold:
                                continue

                            # 2. Check wrist movement & proximity to head
                            for wrist in [left_wrist, right_wrist]:
                                if id_a not in state.previous_wrists:
                                    continue

                                old_wrist = np.array(state.previous_wrists[id_a])
                                movement = distance(wrist, old_wrist)

                                if movement < self.settings.wristMovementThreshold:
                                    continue

                                wrist_to_head = distance(wrist, nose_b)
                                if wrist_to_head < self.settings.wristToHeadThreshold:
                                    if movement > self.settings.strikeMovementThreshold:
                                        violent_frame = True

                            # Save wrist average for next frame
                            avg_wrist = (person_a[9] + person_a[10]) / 2.0
                            state.previous_wrists[id_a] = (float(avg_wrist[0]), float(avg_wrist[1]))
            except Exception as e:
                pass  # Handled safely if keypoints missing box ids

        # 4. Temporal filter update
        if violent_frame:
            state.suspicious_frames += 1
        else:
            state.suspicious_frames = max(0, state.suspicious_frames - 2)

        # 5. Violence status tick
        violence_state = "VIOLENCE DETECTED" if state.suspicious_frames >= self.settings.requiredFrames else "NORMAL"
        tick = ViolenceScoreTick(
            score=min(state.suspicious_frames, self.settings.requiredFrames),
            threshold=self.settings.requiredFrames,
            state=violence_state,
        )

        # 6. Annotate Frame
        color = (0, 0, 255) if violence_state == "VIOLENCE DETECTED" else (0, 255, 0)
        annotated_frame = pose_res.plot()
        annotated_frame = road_res.plot(img=annotated_frame)

        cv2.putText(
            annotated_frame,
            violence_state,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
        )
        cv2.putText(
            annotated_frame,
            f"Score: {state.suspicious_frames}/{self.settings.requiredFrames}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 7. Extract Detection Events — DEDUPLICATED
        #    Only emit the BEST (highest confidence) box per class per frame,
        #    AND only if the per-class cooldown has elapsed.
        events: List[DetectionEvent] = []
        model_names = road_res.names if hasattr(road_res, "names") else {}

        # First pass: find the best box per class on this frame
        best_per_class: Dict[str, Tuple[float, Any]] = {}  # cls -> (conf, box_data)

        if road_res.boxes is not None and len(road_res.boxes) > 0:
            boxes_data = road_res.boxes.data.cpu().numpy()
            for box in boxes_data:
                # box: [x1, y1, x2, y2, conf, cls_id]
                x1, y1, x2, y2, conf, cls_id = box
                cls_name = model_names.get(int(cls_id), "pothole").lower()
                if cls_name not in ["pothole", "accident", "violence"]:
                    cls_name = "pothole"

                # Keep only the highest-confidence detection per class
                if cls_name not in best_per_class or conf > best_per_class[cls_name][0]:
                    best_per_class[cls_name] = (conf, box)

        # Second pass: emit events only for classes that pass cooldown
        for cls_name, (conf, box) in best_per_class.items():
            if not state.can_emit(cls_name, frame_id):
                continue  # Cooldown not elapsed, skip

            x1, y1, x2, y2, conf_val, cls_id = box

            event_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
            )

            # Save static snapshot image if requested
            snapshot_url = None
            if save_snapshot:
                snapshot_filename = f"{event_id}.jpg"
                snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, annotated_frame)
                snapshot_url = f"/static/snapshots/{snapshot_filename}"

            event = DetectionEvent(
                id=event_id,
                type="detection",
                timestamp=timestamp_iso,
                class_name=cls_name,
                confidence=float(conf_val),
                bbox=bbox,
                frameId=frame_id,
                reviewed=False,
                falsePositive=False,
                acknowledged=False,
                snapshotUrl=snapshot_url,
                poseKeypoints=pose_keypoints_all if cls_name == "violence" else None,
            )
            events.append(event)
            state.record_emission(cls_name, frame_id)

        # If violence state triggered and no violence event present, create an explicit event
        # (only if cooldown allows)
        if violence_state == "VIOLENCE DETECTED" and not any(e.class_name == "violence" for e in events):
            if state.can_emit("violence", frame_id):
                event_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
                snapshot_url = None
                if save_snapshot:
                    snapshot_filename = f"{event_id}.jpg"
                    snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
                    cv2.imwrite(snapshot_path, annotated_frame)
                    snapshot_url = f"/static/snapshots/{snapshot_filename}"

                h, w, _ = frame.shape
                bbox = BoundingBox(x=0, y=0, w=float(w), h=float(h))
                event = DetectionEvent(
                    id=event_id,
                    type="detection",
                    timestamp=timestamp_iso,
                    class_name="violence",
                    confidence=0.90,
                    bbox=bbox,
                    frameId=frame_id,
                    reviewed=False,
                    falsePositive=False,
                    acknowledged=False,
                    snapshotUrl=snapshot_url,
                    poseKeypoints=pose_keypoints_all,
                )
                events.append(event)
                state.record_emission("violence", frame_id)

        return annotated_frame, events, tick
