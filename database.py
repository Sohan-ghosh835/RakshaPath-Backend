"""
SQLite Database Storage Layer for Events, Settings, Analytics, and Video Processing Jobs
"""

import os
import json
import sqlite3
import datetime
from typing import List, Dict, Optional, Tuple, Any
try:
    from backend.schemas import (
        DetectionEvent,
        BoundingBox,
        PoseKeypoint,
        AppSettings,
        AnalyticsSummary,
        TimeSeriesDataPoint,
        HeatmapCell,
        VideoJobResponse,
    )
except ImportError:
    from schemas import (
        DetectionEvent,
        BoundingBox,
        PoseKeypoint,
        AppSettings,
        AnalyticsSummary,
        TimeSeriesDataPoint,
        HeatmapCell,
        VideoJobResponse,
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "roadwatch.db")


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_w REAL NOT NULL,
                    bbox_h REAL NOT NULL,
                    frame_id INTEGER NOT NULL,
                    reviewed INTEGER DEFAULT 0,
                    false_positive INTEGER DEFAULT 0,
                    acknowledged INTEGER DEFAULT 0,
                    snapshot_url TEXT,
                    pose_keypoints TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_cls ON events(class_name)")

            # Video jobs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    total_frames INTEGER DEFAULT 0,
                    processed_frames INTEGER DEFAULT 0,
                    fps REAL DEFAULT 0.0,
                    duration_seconds REAL DEFAULT 0.0,
                    output_url TEXT,
                    events_count INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    results_json TEXT
                )
                """
            )

            # Settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            conn.commit()

    def add_event(self, event: DetectionEvent):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            kpts_json = json.dumps([k.dict() for sublist in event.poseKeypoints for k in sublist]) if event.poseKeypoints else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO events (
                    id, timestamp, class_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    frame_id, reviewed, false_positive, acknowledged,
                    snapshot_url, pose_keypoints
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.class_name,
                    event.confidence,
                    event.bbox.x,
                    event.bbox.y,
                    event.bbox.w,
                    event.bbox.h,
                    event.frameId,
                    1 if event.reviewed else 0,
                    1 if event.falsePositive else 0,
                    1 if event.acknowledged else 0,
                    event.snapshotUrl,
                    kpts_json,
                ),
            )
            conn.commit()

    def _row_to_event(self, row: sqlite3.Row) -> DetectionEvent:
        return DetectionEvent(
            id=row["id"],
            type="detection",
            timestamp=row["timestamp"],
            class_name=row["class_name"],
            confidence=row["confidence"],
            bbox=BoundingBox(
                x=row["bbox_x"],
                y=row["bbox_y"],
                w=row["bbox_w"],
                h=row["bbox_h"],
            ),
            frameId=row["frame_id"],
            reviewed=bool(row["reviewed"]),
            falsePositive=bool(row["false_positive"]),
            acknowledged=bool(row["acknowledged"]),
            snapshotUrl=row["snapshot_url"],
        )

    def get_events(
        self,
        types: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        unreviewed_only: bool = False,
        min_confidence: float = 0.0,
    ) -> Tuple[List[DetectionEvent], int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            conditions = ["1=1"]
            params: List[Any] = []

            if types and len(types) > 0:
                placeholders = ",".join("?" for _ in types)
                conditions.append(f"class_name IN ({placeholders})")
                params.extend(types)

            if from_date:
                conditions.append("timestamp >= ?")
                params.append(from_date)

            if to_date:
                conditions.append("timestamp <= ?")
                params.append(to_date)

            if unreviewed_only:
                conditions.append("reviewed = 0")

            if min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(min_confidence)

            where_clause = " AND ".join(conditions)

            # Count total
            count_sql = f"SELECT COUNT(*) FROM events WHERE {where_clause}"
            cursor.execute(count_sql, params)
            total = cursor.fetchone()[0]

            # Query results
            query_sql = f"""
                SELECT * FROM events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(query_sql, params + [limit, offset])
            rows = cursor.fetchall()
            events = [self._row_to_event(r) for r in rows]

            return events, total

    def get_event(self, event_id: str) -> Optional[DetectionEvent]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_event(row)
            return None

    def update_event(self, event_id: str, updates: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = []
            params = []

            if "reviewed" in updates:
                fields.append("reviewed = ?")
                params.append(1 if updates["reviewed"] else 0)
            if "falsePositive" in updates:
                fields.append("false_positive = ?")
                params.append(1 if updates["falsePositive"] else 0)
            if "acknowledged" in updates:
                fields.append("acknowledged = ?")
                params.append(1 if updates["acknowledged"] else 0)

            if fields:
                params.append(event_id)
                sql = f"UPDATE events SET {', '.join(fields)} WHERE id = ?"
                cursor.execute(sql, params)
                conn.commit()

    def save_job(self, job: VideoJobResponse):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            results_str = None
            if job.events or job.violenceCurve:
                results_str = json.dumps({
                    "events": [e.dict(by_alias=True) for e in job.events] if job.events else [],
                    "violenceCurve": job.violenceCurve or []
                })

            cursor.execute(
                """
                INSERT OR REPLACE INTO video_jobs (
                    job_id, filename, status, progress, total_frames,
                    processed_frames, fps, duration_seconds, output_url,
                    events_count, error, created_at, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.jobId,
                    job.filename,
                    job.status,
                    job.progress,
                    job.totalFrames,
                    job.processedFrames,
                    job.fps,
                    job.durationSeconds,
                    job.outputUrl,
                    job.eventsCount,
                    job.error,
                    datetime.datetime.utcnow().isoformat() + "Z",
                    results_str,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[VideoJobResponse]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM video_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None

            events = None
            violence_curve = None
            if row["results_json"]:
                try:
                    data = json.loads(row["results_json"])
                    raw_events = data.get("events", [])
                    events = []
                    for e in raw_events:
                        if isinstance(e, dict):
                            if "class" in e and "class_name" not in e:
                                e["class_name"] = e["class"]
                            try:
                                events.append(DetectionEvent(**e))
                            except Exception:
                                pass
                    violence_curve = data.get("violenceCurve")
                except Exception as ex:
                    print(f"[Database] Warning parsing results_json for {job_id}: {ex}")

            return VideoJobResponse(
                jobId=row["job_id"],
                filename=row["filename"],
                status=row["status"],
                progress=row["progress"],
                totalFrames=row["total_frames"],
                processedFrames=row["processed_frames"],
                fps=row["fps"],
                durationSeconds=row["duration_seconds"],
                outputUrl=row["output_url"],
                eventsCount=row["events_count"],
                error=row["error"],
                events=events,
                violenceCurve=violence_curve,
            )

    def get_analytics_summary(self) -> AnalyticsSummary:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            today_str = datetime.date.today().isoformat()

            cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (today_str,))
            today_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM events")
            total_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT class_name, COUNT(*) FROM events GROUP BY class_name"
            )
            by_class_rows = cursor.fetchall()
            by_class: Dict[str, int] = {"pothole": 0, "accident": 0, "violence": 0}
            for row in by_class_rows:
                cls_name = row[0]
                if cls_name in by_class:
                    by_class[cls_name] = row[1]

            cursor.execute("SELECT COUNT(*) FROM events WHERE false_positive = 1")
            fp_count = cursor.fetchone()[0]
            fp_rate = (fp_count / total_count * 100.0) if total_count > 0 else 0.0

            return AnalyticsSummary(
                totalEventsToday=today_count,
                totalEventsWeek=total_count,
                byClass=by_class,
                averageViolenceScore=2.4,
                falsePositiveRate=round(fp_rate, 1),
            )

    def get_analytics_timeseries(self, window: str = "day") -> List[TimeSeriesDataPoint]:
        points: List[TimeSeriesDataPoint] = []
        buckets = 24 if window == "day" else 7 if window == "week" else 30
        now = datetime.datetime.utcnow()

        for i in range(buckets - 1, -1, -1):
            if window == "day":
                dt = now - datetime.timedelta(hours=i)
                ts_str = dt.strftime("%Y-%m-%dT%H:00:00Z")
            else:
                dt = now - datetime.timedelta(days=i)
                ts_str = dt.strftime("%Y-%m-%dT00:00:00Z")

            points.append(
                TimeSeriesDataPoint(
                    timestamp=ts_str,
                    pothole=max(0, (i * 3 + 2) % 7),
                    accident=max(0, (i * 2 + 1) % 4),
                    violence=1 if i % 5 == 0 else 0,
                )
            )
        return points

    def get_analytics_heatmap(self) -> List[HeatmapCell]:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cells: List[HeatmapCell] = []
        for d_idx, day in enumerate(days):
            for hour in range(24):
                count = (d_idx * 2 + hour * 3) % 6
                cells.append(HeatmapCell(day=day, hour=hour, count=count))
        return cells

    def get_settings(self) -> AppSettings:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'app_settings'")
            row = cursor.fetchone()
            if row:
                try:
                    return AppSettings(**json.loads(row[0]))
                except Exception:
                    pass
            return AppSettings()

    def update_settings(self, settings: AppSettings) -> AppSettings:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            settings_json = json.dumps(settings.dict(by_alias=True))
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('app_settings', ?)",
                (settings_json,),
            )
            conn.commit()
        return settings
