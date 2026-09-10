"""
Pydantic data schemas matching API_CONTRACT.md
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# --- Detection Classes ---
DetectionClass = Literal["pothole", "accident", "violence"]


# --- Bounding Box ---
class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


# --- Keypoint ---
class PoseKeypoint(BaseModel):
    x: float
    y: float
    confidence: float


# --- Detection Event ---
class DetectionEvent(BaseModel):
    id: str
    type: Literal["detection"] = "detection"
    timestamp: str  # ISO 8601
    class_name: DetectionClass = Field(..., alias="class")
    confidence: float
    bbox: BoundingBox
    frameId: int
    reviewed: bool = False
    falsePositive: bool = False
    acknowledged: bool = False
    snapshotUrl: Optional[str] = None
    poseKeypoints: Optional[List[List[PoseKeypoint]]] = None

    class Config:
        populate_by_name = True


# --- Violence Score Tick ---
class ViolenceScoreTick(BaseModel):
    type: Literal["violence_score"] = "violence_score"
    score: int
    threshold: int = 6
    state: Literal["NORMAL", "VIOLENCE DETECTED"] = "NORMAL"


# --- Settings ---
class DetectionSettings(BaseModel):
    detectionConfidence: float = 0.50
    poseConfidence: float = 0.50
    requiredFrames: int = 6
    wristMovementThreshold: float = 18.0
    wristToHeadThreshold: float = 150.0
    personDistanceThreshold: float = 350.0
    strikeMovementThreshold: float = 30.0


class NotificationSettings(BaseModel):
    soundEnabled: bool = True
    soundVolume: float = 0.8
    browserPushEnabled: bool = False
    webhookUrl: str = ""
    emailUrl: str = ""
    notifyOnViolence: bool = True
    notifyOnAccident: bool = True


class AppearanceSettings(BaseModel):
    theme: Literal["dark", "light"] = "dark"
    reduceMotion: bool = False


class CameraSource(BaseModel):
    id: str
    name: str
    source: str  # URL or device index
    isDefault: bool = False


class AppSettings(BaseModel):
    detection: DetectionSettings = DetectionSettings()
    notifications: NotificationSettings = NotificationSettings()
    appearance: AppearanceSettings = AppearanceSettings()
    cameras: List[CameraSource] = [
        CameraSource(id="cam-1", name="Camera 01 - Main Road", source="0", isDefault=True)
    ]


# --- Analytics ---
class AnalyticsSummary(BaseModel):
    totalEventsToday: int
    totalEventsWeek: int
    byClass: Dict[str, int]
    averageViolenceScore: float
    falsePositiveRate: float


class TimeSeriesDataPoint(BaseModel):
    timestamp: str
    pothole: int
    accident: int
    violence: int


class HeatmapCell(BaseModel):
    day: str
    hour: int
    count: int


# --- Video Upload & Processing Job ---
class VideoJobResponse(BaseModel):
    jobId: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: float = 0.0  # 0.0 to 100.0
    totalFrames: int = 0
    processedFrames: int = 0
    fps: float = 0.0
    durationSeconds: float = 0.0
    outputUrl: Optional[str] = None
    eventsCount: int = 0
    error: Optional[str] = None
    events: Optional[List[DetectionEvent]] = None
    violenceCurve: Optional[List[Dict[str, Any]]] = None


# --- Event Review Update Request ---
class EventUpdateRequest(BaseModel):
    reviewed: Optional[bool] = None
    falsePositive: Optional[bool] = None
    acknowledged: Optional[bool] = None
