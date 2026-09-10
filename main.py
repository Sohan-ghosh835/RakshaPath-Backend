"""
FastAPI Backend Application Entry Point — Road Issues Detection Service
Provides WebSocket event streaming, MJPEG video feed, Video Upload processing (Mode A),
and REST APIs matching API_CONTRACT.md.
"""

import os
import uuid
import asyncio
import shutil
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Query,
    HTTPException,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from backend.schemas import (
        DetectionEvent,
        AppSettings,
        AnalyticsSummary,
        TimeSeriesDataPoint,
        HeatmapCell,
        VideoJobResponse,
        EventUpdateRequest,
    )
    from backend.database import Database
    from backend.detector import RoadDetectionEngine
    from backend.camera_manager import WebcamManager
    from backend.video_processor import process_video_job_async
except ImportError:
    from schemas import (
        DetectionEvent,
        AppSettings,
        AnalyticsSummary,
        TimeSeriesDataPoint,
        HeatmapCell,
        VideoJobResponse,
        EventUpdateRequest,
    )
    from database import Database
    from detector import RoadDetectionEngine
    from camera_manager import WebcamManager
    from video_processor import process_video_job_async

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
SNAPSHOTS_DIR = os.path.join(STATIC_DIR, "snapshots")

for d in [STATIC_DIR, UPLOADS_DIR, VIDEOS_DIR, SNAPSHOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize Database & Detection Engine
db = Database()
engine = RoadDetectionEngine(
    pose_model_path=os.path.join(BASE_DIR, "yolo11n-pose.pt"),
    road_model_path=os.path.join(BASE_DIR, "Road Yolo.pt"),
    snapshot_dir=SNAPSHOTS_DIR,
    settings=db.get_settings().detection,
)
camera_manager = WebcamManager(engine=engine, db=db, camera_index=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start live webcam capture loop in background
    capture_task = asyncio.create_task(camera_manager.capture_loop())
    print("[FastAPI] Background webcam capture loop started.")
    yield
    # Shutdown: stop camera
    camera_manager.stop()
    capture_task.cancel()
    print("[FastAPI] Backend shutdown complete.")


app = FastAPI(
    title="Road Issues & Violence Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mount
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- WebSocket Endpoint ---
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await camera_manager.register_websocket(websocket)
    try:
        while True:
            # Keep connection alive & listen for client messages
            msg = await websocket.receive_text()
    except WebSocketDisconnect:
        camera_manager.unregister_websocket(websocket)
    except Exception:
        camera_manager.unregister_websocket(websocket)


# --- MJPEG Live Video Stream Endpoint ---
@app.get("/stream/mjpeg")
async def stream_mjpeg():
    async def generate_frames():
        while True:
            jpeg = camera_manager.get_mjpeg_frame()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- Mode A: Video Upload & Processing Endpoints ---
@app.post("/api/v1/video/upload", response_model=VideoJobResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: .mp4, .avi, .mov",
        )

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    upload_filename = f"{job_id}{ext}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = VideoJobResponse(
        jobId=job_id,
        filename=file.filename,
        status="queued",
        progress=0.0,
    )
    db.save_job(job)

    # Launch processing in background task
    background_tasks.add_task(
        process_video_job_async,
        engine=engine,
        db=db,
        job_id=job_id,
        input_path=upload_path,
        output_dir=VIDEOS_DIR,
    )

    return job


@app.get("/api/v1/video/jobs/{job_id}", response_model=VideoJobResponse)
async def get_video_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video processing job not found")
    return job


@app.get("/api/v1/video/download/{job_id}")
async def download_video(job_id: str):
    job = db.get_job(job_id)
    if not job or not job.outputUrl:
        raise HTTPException(status_code=404, detail="Processed video not ready")

    filename = os.path.basename(job.outputUrl)
    file_path = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file missing on server")

    return FileResponse(file_path, media_type="video/mp4", filename=f"processed_{job.filename}")


# --- REST API Endpoints (API_CONTRACT.md) ---

@app.get("/api/v1/events")
async def get_events(
    type: Optional[str] = Query(None, description="Comma separated classes: pothole,accident,violence"),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    unreviewed_only: bool = Query(False),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
):
    types_list = [t.strip() for t in type.split(",")] if type else None
    events, total = db.get_events(
        types=types_list,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
        unreviewed_only=unreviewed_only,
        min_confidence=min_confidence,
    )

    return {
        "events": [e.dict(by_alias=True) for e in events],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    event = db.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event.dict(by_alias=True)


@app.put("/api/v1/events/{event_id}")
async def update_event(event_id: str, updates: EventUpdateRequest):
    event = db.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    db.update_event(event_id, updates.dict(exclude_unset=True))
    updated = db.get_event(event_id)
    return updated.dict(by_alias=True)


@app.get("/api/v1/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary():
    return db.get_analytics_summary()


@app.get("/api/v1/analytics/timeseries")
async def get_analytics_timeseries(window: str = Query("day", pattern="^(day|week|month)$")):
    data = db.get_analytics_timeseries(window=window)
    return {"data": [d.dict() for d in data]}


@app.get("/api/v1/analytics/heatmap")
async def get_analytics_heatmap():
    data = db.get_analytics_heatmap()
    return {"data": [d.dict() for d in data]}


@app.get("/api/v1/settings", response_model=AppSettings)
async def get_settings():
    return db.get_settings()


@app.put("/api/v1/settings", response_model=AppSettings)
async def update_settings(settings: AppSettings):
    updated = db.update_settings(settings)
    engine.update_settings(updated.detection)
    return updated


@app.get("/api/v1/system/info")
async def get_system_info():
    return {
        "status": "online",
        "device": engine.device.upper(),
        "cuda_available": engine.device == "cuda",
        "models": {
            "pose": "yolo11n-pose.pt",
            "road": "Road Yolo.pt",
        },
        "camera_active": camera_manager.cap is not None and camera_manager.cap.isOpened(),
        "active_websockets": len(camera_manager.active_websockets),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
