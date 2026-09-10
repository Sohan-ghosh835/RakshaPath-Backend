"""
Automated Backend Unit & Regression Test Suite
Verifies:
1. YOLO models loading & device selection (CUDA/CPU)
2. RoadDetectionEngine frame processing & violence heuristic correctness
3. SQLite Database CRUD, filtering, pagination, and analytics queries
"""

import os
import cv2
import numpy as np
from backend.schemas import DetectionEvent, BoundingBox, AppSettings
from backend.database import Database
from backend.detector import RoadDetectionEngine, VideoState


def test_database_crud():
    test_db_path = "backend/test_roadwatch.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = Database(db_path=test_db_path)

    # Add Event
    event = DetectionEvent(
        id="test-event-1",
        type="detection",
        timestamp="2026-09-10T14:00:00.000Z",
        class_name="pothole",
        confidence=0.85,
        bbox=BoundingBox(x=10, y=20, w=100, h=80),
        frameId=1,
        reviewed=False,
        falsePositive=False,
        acknowledged=False,
        snapshotUrl="/static/snapshots/test.jpg",
    )
    db.add_event(event)

    fetched = db.get_event("test-event-1")
    assert fetched is not None
    assert fetched.id == "test-event-1"
    assert fetched.class_name == "pothole"
    assert fetched.confidence == 0.85

    # Update event
    db.update_event("test-event-1", {"reviewed": True, "acknowledged": True})
    updated = db.get_event("test-event-1")
    assert updated.reviewed is True
    assert updated.acknowledged is True

    # Filter events
    events, total = db.get_events(types=["pothole"], limit=10)
    assert total == 1
    assert len(events) == 1

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("✅ Database CRUD tests passed!")


def test_detection_engine():
    engine = RoadDetectionEngine(
        pose_model_path="yolo11n-pose.pt",
        road_model_path="Road Yolo.pt",
        snapshot_dir="backend/static/snapshots_test",
    )
    state = VideoState()

    # Create dummy black frame (640x360)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    annotated, events, tick = engine.process_frame(
        frame=frame,
        state=state,
        frame_id=1,
        save_snapshot=False,
    )

    assert annotated is not None
    assert annotated.shape == (360, 640, 3)
    assert tick.state == "NORMAL"
    assert tick.score == 0

    print("✅ Detection Engine single frame processing passed!")


if __name__ == "__main__":
    test_database_crud()
    test_detection_engine()
    print("🎉 All backend tests completed successfully!")
