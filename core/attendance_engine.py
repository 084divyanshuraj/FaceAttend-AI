"""FaceAttend AI - Real-Time Attendance Engine & Temporal Stabilization State Machine.

Coordinates multi-face detection, landmark alignment, deep embedding extraction,
in-memory vector matching, sliding-window consensus filtering, and idempotent attendance logging.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

import config
from core.detector import FaceDetection, YuNetDetector
from core.matcher import GLOBAL_MATCHER_CACHE, MatchResult, VectorMatcherCache
from core.recognizer import SFaceRecognizer
from database import db


@dataclass
class TrackedFaceResult:
    """Individual face detection and recognition result within a frame."""
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    landmarks: List[List[float]]         # [[x, y], ...] 5 points
    student_id: Optional[int]
    roll_number: Optional[str]
    name: Optional[str]
    department: Optional[str]
    confidence: float
    status: str                          # "NEWLY_MARKED", "ALREADY_MARKED", "ACQUIRING", "UNKNOWN"
    badge_color: str                     # Hex color code for UI rendering
    label: str                           # User-facing label

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedFrameResult:
    """Consolidated result for an analyzed video frame."""
    timestamp: str
    faces_detected: int
    faces: List[TrackedFaceResult]
    new_attendance_events: List[Dict[str, Any]]
    session_stats: Dict[str, Any]
    annotated_frame: Optional[np.ndarray] = None


class AttendanceEngine:
    """Real-time multi-face attendance engine with temporal consensus and cooldown."""

    # UI Theme Palette
    COLOR_MARKED = "#10B981"      # Emerald green (attendance confirmed / present)
    COLOR_ACQUIRING = "#F59E0B"   # Amber (verifying consensus across frames)
    COLOR_UNKNOWN = "#EF4444"     # Crimson red (unmatched face)

    def __init__(
        self,
        detector: Optional[YuNetDetector] = None,
        recognizer: Optional[SFaceRecognizer] = None,
        matcher_cache: Optional[VectorMatcherCache] = None,
        window_size: int = config.ATTENDANCE_CONFIG["temporal_window_size"],
        consensus_threshold: int = config.ATTENDANCE_CONFIG["temporal_consensus_count"],
        cooldown_seconds: float = config.ATTENDANCE_CONFIG["session_cooldown_seconds"],
    ):
        self.detector = detector or YuNetDetector()
        self.recognizer = recognizer or SFaceRecognizer()
        self.matcher = matcher_cache or GLOBAL_MATCHER_CACHE

        self.window_size = int(window_size)
        self.consensus_threshold = int(consensus_threshold)
        self.cooldown_seconds = float(cooldown_seconds)

        # Sliding window history buffer: student_id -> deque of timestamps
        self._history: Dict[int, deque] = {}

        # Cache of student_ids already marked present in the active session
        self._marked_students: Set[int] = set()
        self._last_active_session_id: Optional[int] = None

        # Cooldown tracker: student_id -> last_marked_timestamp
        self._last_marked_time: Dict[int, float] = {}

    def sync_session_state(
        self,
        session_id: int,
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Sync in-memory presence cache with SQLite for the active session."""
        if self._last_active_session_id != session_id:
            self._marked_students.clear()
            self._history.clear()
            self._last_marked_time.clear()
            self._last_active_session_id = session_id

            # Load students already marked present in this session
            records = db.get_session_attendance(session_id, db_path=db_path)
            for r in records:
                self._marked_students.add(r["student_id"])

    def process_frame(
        self,
        frame: np.ndarray,
        session_id: Optional[int] = None,
        draw_annotations: bool = False,
        db_path: Optional[Union[str, Path]] = None,
    ) -> ProcessedFrameResult:
        """Process a single camera frame for multi-face detection and attendance logging.

        Args:
            frame: Full BGR camera image.
            session_id: Optional active session ID. If None, performs live recognition without logging.
            draw_annotations: If True, draws bounding boxes and text directly onto frame copy.
            db_path: Optional custom database path.

        Returns:
            ProcessedFrameResult containing face tracks, newly marked events, and session stats.
        """
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now_epoch = time.time()

        if session_id:
            self.sync_session_state(session_id, db_path=db_path)

        # 1. Detect all faces in frame
        detections = self.detector.detect(frame)

        tracked_faces: List[TrackedFaceResult] = []
        new_events: List[Dict[str, Any]] = []

        # Annotated frame copy if requested
        annotated = frame.copy() if draw_annotations else None

        for det in detections:
            # 2. Align face and extract 128-D embedding
            try:
                aligned = self.recognizer.align_crop(frame, det)
                embedding = self.recognizer.extract_feature(aligned, normalize=True)
                match: MatchResult = self.matcher.match(embedding)
            except Exception:
                # If alignment/crop fails on boundary edge, treat as unknown
                match = MatchResult(
                    is_known=False,
                    student_id=None,
                    roll_number=None,
                    name=None,
                    department=None,
                    confidence=0.0,
                    label="Unknown",
                )

            # Determine attendance status and badge
            if not match.is_known or match.student_id is None:
                # Unknown Person
                status = "UNKNOWN"
                badge_color = self.COLOR_UNKNOWN
                label = "Unknown"
            else:
                student_id = match.student_id

                # Check if already marked present in this active session
                if session_id and student_id in self._marked_students:
                    status = "ALREADY_MARKED"
                    badge_color = self.COLOR_MARKED
                    label = f"{match.name} - Present"
                else:
                    # Update sliding window history
                    if student_id not in self._history:
                        self._history[student_id] = deque(maxlen=self.window_size)
                    self._history[student_id].append(now_epoch)

                    # Purge records older than 3 seconds to avoid stale accumulation
                    recent_hits = [
                        t for t in self._history[student_id] if (now_epoch - t) <= 3.0
                    ]
                    self._history[student_id] = deque(recent_hits, maxlen=self.window_size)

                    # Evaluate consensus threshold
                    if len(recent_hits) >= self.consensus_threshold:
                        # Check cooldown window
                        last_marked = self._last_marked_time.get(student_id, 0.0)
                        if (now_epoch - last_marked) >= self.cooldown_seconds:
                            status = "NEWLY_MARKED"
                            badge_color = self.COLOR_MARKED
                            label = f"{match.name} - Present"

                            # Mark in SQLite if an active session exists
                            if session_id:
                                is_new, att_id = db.mark_attendance(
                                    session_id=session_id,
                                    student_id=student_id,
                                    confidence_score=match.confidence,
                                    method="AUTO_FACE",
                                    remarks=f"Confidence: {match.confidence:.2f}",
                                    db_path=db_path,
                                )
                                if is_new:
                                    self._marked_students.add(student_id)
                                    self._last_marked_time[student_id] = now_epoch
                                    new_events.append({
                                        "attendance_id": att_id,
                                        "student_id": student_id,
                                        "roll_number": match.roll_number,
                                        "name": match.name,
                                        "department": match.department,
                                        "confidence": round(match.confidence, 3),
                                        "timestamp": now_ts,
                                    })
                        else:
                            status = "ALREADY_MARKED"
                            badge_color = self.COLOR_MARKED
                            label = f"{match.name} - Present"
                    else:
                        # Consensus acquiring
                        status = "ACQUIRING"
                        badge_color = self.COLOR_ACQUIRING
                        label = f"Verifying {match.name}... ({len(recent_hits)}/{self.consensus_threshold})"

            # Landmark coordinates for canvas/rendering
            landmarks_list = det.landmarks.tolist() if det.landmarks is not None else []

            tracked_face = TrackedFaceResult(
                bbox=det.bbox,
                landmarks=landmarks_list,
                student_id=match.student_id,
                roll_number=match.roll_number,
                name=match.name,
                department=match.department,
                confidence=round(match.confidence, 3),
                status=status,
                badge_color=badge_color,
                label=label,
            )
            tracked_faces.append(tracked_face)

            # Draw visual bounding box and label if requested
            if draw_annotations and annotated is not None:
                self._draw_face_overlay(annotated, tracked_face)

        # Session summary metrics
        stats = db.get_dashboard_stats(session_id=session_id, db_path=db_path) if session_id else {
            "total_students": len(self.matcher),
            "present_students": len(self._marked_students),
            "absent_students": 0,
            "attendance_percentage": 0.0,
        }

        return ProcessedFrameResult(
            timestamp=now_ts,
            faces_detected=len(tracked_faces),
            faces=tracked_faces,
            new_attendance_events=new_events,
            session_stats=stats,
            annotated_frame=annotated,
        )

    def _draw_face_overlay(self, image: np.ndarray, face: TrackedFaceResult) -> None:
        """Render modern bounding box, semi-transparent label badge, and text."""
        x, y, w, h = face.bbox
        # Convert hex color to BGR for OpenCV
        hex_col = face.badge_color.lstrip("#")
        r, g, b = tuple(int(hex_col[i:i + 2], 16) for i in (0, 2, 4))
        bgr_color = (b, g, r)

        # 1. Rounded-corner or clean solid bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, 2)

        # 2. Text Label
        text = f"{face.label} ({face.confidence:.0%})" if face.confidence > 0 else face.label
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        badge_y1 = max(0, y - th - 12)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = x + tw + 14

        # Background badge for text
        cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), bgr_color, -1)
        # White text label
        cv2.putText(
            image,
            text,
            (x + 7, y - 6),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

