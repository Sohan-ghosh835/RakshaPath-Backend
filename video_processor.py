"""
Mode A — Async Video File Processing Engine
Processes uploaded video files (.mp4, .avi, .mov) frame-by-frame with YOLO,
writes annotated output video files (H.264 via imageio-ffmpeg), and records
timestamp-synced detection events.
"""

import os
import cv2
import time
import datetime
import asyncio
import imageio.v2 as iio
from typing import List, Dict, Any, Optional
try:
    from backend.detector import RoadDetectionEngine, VideoState
    from backend.database import Database
    from backend.schemas import VideoJobResponse, DetectionEvent, BoundingBox
except ImportError:
    from detector import RoadDetectionEngine, VideoState
    from database import Database
    from schemas import VideoJobResponse, DetectionEvent, BoundingBox


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEOS_DIR = os.path.join(BASE_DIR, "static", "videos")


async def process_video_job_async(
    engine: RoadDetectionEngine,
    db: Database,
    job_id: str,
    input_path: str,
    output_dir: Optional[str] = None,
):
    """Executes video file processing in a background thread."""
    if output_dir is None:
        output_dir = DEFAULT_VIDEOS_DIR
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _process_video_job_sync,
        engine,
        db,
        job_id,
        input_path,
        output_dir,
    )


def _process_video_job_sync(
    engine: RoadDetectionEngine,
    db: Database,
    job_id: str,
    input_path: str,
    output_dir: str,
):
    if not output_dir:
        output_dir = DEFAULT_VIDEOS_DIR
    os.makedirs(output_dir, exist_ok=True)

    job = db.get_job(job_id)
    if not job:
        return

    job.status = "processing"
    db.save_job(job)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        job.status = "failed"
        job.error = "Could not open video file."
        db.save_job(job)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    duration_seconds = total_frames / fps

    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Use imageio-ffmpeg for browser-compatible H.264 output
    writer = iio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=None,
        bitrate=None,
        output_params=[
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )

    # Isolated per-job video state
    state = VideoState(required_frames=engine.settings.requiredFrames)

    all_events: List[DetectionEvent] = []
    violence_curve: List[Dict[str, Any]] = []

    start_timestamp = datetime.datetime.utcnow()
    frame_idx = 0

    print(f"[VideoProcessor] Starting processing for job {job_id} ({total_frames} frames, {fps:.1f} FPS)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_time_sec = frame_idx / fps
            frame_iso = (start_timestamp + datetime.timedelta(seconds=frame_time_sec)).isoformat() + "Z"

            annotated_frame, events, tick = engine.process_frame(
                frame=frame,
                state=state,
                frame_id=frame_idx,
                timestamp_iso=frame_iso,
                save_snapshot=True,
            )

            # Write annotated frame (convert BGR→RGB for imageio)
            writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

            # Collect events (already deduplicated by the engine's cooldown logic)
            for ev in events:
                db.add_event(ev)
                all_events.append(ev)

            # Collect violence score datapoint
            if frame_idx % int(max(1, fps / 5)) == 0 or tick.state == "VIOLENCE DETECTED":
                violence_curve.append({
                    "timeSec": round(frame_time_sec, 2),
                    "frame": frame_idx,
                    "score": tick.score,
                    "state": tick.state,
                })

            # Update job progress
            progress = min(99.0, (frame_idx / total_frames) * 100.0)
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                job.progress = round(progress, 1)
                job.processedFrames = frame_idx
                job.totalFrames = total_frames
                job.fps = round(fps, 1)
                job.durationSeconds = round(duration_seconds, 1)
                db.save_job(job)

    except Exception as e:
        print(f"[VideoProcessor] Error processing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        job.status = "failed"
        job.error = str(e)
        db.save_job(job)
        cap.release()
        writer.close()
        return

    cap.release()
    writer.close()

    job.status = "completed"
    job.progress = 100.0
    job.processedFrames = total_frames
    job.totalFrames = total_frames
    job.fps = round(fps, 1)
    job.durationSeconds = round(duration_seconds, 1)
    job.outputUrl = f"/static/videos/{output_filename}"
    job.eventsCount = len(all_events)
    job.events = all_events
    job.violenceCurve = violence_curve

    db.save_job(job)
    print(f"[VideoProcessor] Job {job_id} completed. {len(all_events)} events detected.")
