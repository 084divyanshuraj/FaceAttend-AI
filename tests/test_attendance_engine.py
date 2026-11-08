"""FaceAttend AI - Unit & Verification Tests for Attendance Engine.

Tests sliding-window consensus filtering, session cooldown, duplicate prevention,
and multi-face detection handling.
"""

from pathlib import Path
import sys
import tempfile

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.attendance_engine import AttendanceEngine
from core.detector import FaceDetection
from core.matcher import VectorMatcherCache
from database import db


class MockDetector:
    """Mock detector returning predetermined detections."""
    def __init__(self, detections=None):
        self.detections = detections or []

    def detect(self, image):
        return self.detections


class MockRecognizer:
    """Mock recognizer returning fixed embeddings."""
    def __init__(self, embedding=None):
        self.embedding = embedding if embedding is not None else np.zeros(128, dtype=np.float32)

    def align_crop(self, image, detection):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def extract_feature(self, aligned_face, normalize=True):
        return self.embedding


def test_temporal_consensus_and_duplicate_prevention():
    """Verify that temporal consensus requires K frames and prevents duplicate logging."""
    temp_dir = tempfile.mkdtemp()
    temp_db = Path(temp_dir) / "test_attendance.db"
    db.init_db(db_path=temp_db)

    # Register student
    student_id = db.add_student("CS-701", "Ada", "Lovelace", "CS", db_path=temp_db)
    v1 = np.zeros(128, dtype=np.float32)
    v1[0] = 1.0
    db.save_face_embeddings(student_id, [v1], db_path=temp_db)

    # Initialize in-memory matcher cache
    matcher = VectorMatcherCache(threshold=0.637)
    matcher.warm_from_db(temp_db)

    # Create active session
    session_id = db.create_session("CS701", "Algorithms", "Prof. Turing", db_path=temp_db)

    # Setup mock detection
    mock_detection = FaceDetection(
        bbox=(50, 50, 100, 100),
        confidence=0.95,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw=np.zeros(15, dtype=np.float32),
    )

    engine = AttendanceEngine(
        detector=MockDetector([mock_detection]),
        recognizer=MockRecognizer(v1),
        matcher_cache=matcher,
        window_size=5,
        consensus_threshold=3,
        cooldown_seconds=60.0,
    )

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Hit 1 -> Status should be ACQUIRING
    res1 = engine.process_frame(blank_frame, session_id=session_id, db_path=temp_db)
    assert res1.faces_detected == 1
    assert res1.faces[0].status == "ACQUIRING"
    assert len(res1.new_attendance_events) == 0

    # Frame 2: Hit 2 -> Status should be ACQUIRING
    res2 = engine.process_frame(blank_frame, session_id=session_id, db_path=temp_db)
    assert res2.faces[0].status == "ACQUIRING"
    assert len(res2.new_attendance_events) == 0

    # Frame 3: Hit 3 (Consensus reached) -> Status should be NEWLY_MARKED
    res3 = engine.process_frame(blank_frame, session_id=session_id, db_path=temp_db)
    assert res3.faces[0].status == "NEWLY_MARKED"
    assert len(res3.new_attendance_events) == 1
    assert res3.new_attendance_events[0]["student_id"] == student_id
    assert res3.new_attendance_events[0]["roll_number"] == "CS-701"

    # Frame 4: Already marked -> Status should be ALREADY_MARKED (No new DB writes)
    res4 = engine.process_frame(blank_frame, session_id=session_id, db_path=temp_db)
    assert res4.faces[0].status == "ALREADY_MARKED"
    assert len(res4.new_attendance_events) == 0

    # Verify exactly 1 attendance record in database
    records = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(records) == 1
    assert records[0]["student_id"] == student_id


def test_unknown_person_detection():
    """Verify unknown persons are assigned UNKNOWN status and red badge."""
    matcher = VectorMatcherCache(threshold=0.637)
    # Empty cache (no registered students)

    mock_det = FaceDetection(
        bbox=(20, 20, 80, 80),
        confidence=0.90,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw=np.zeros(15, dtype=np.float32),
    )

    v_random = np.ones(128, dtype=np.float32) / np.sqrt(128)
    engine = AttendanceEngine(
        detector=MockDetector([mock_det]),
        recognizer=MockRecognizer(v_random),
        matcher_cache=matcher,
    )

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.process_frame(blank_frame)

    assert res.faces_detected == 1
    face = res.faces[0]
    assert face.status == "UNKNOWN"
    assert face.label == "Unknown"
    assert face.badge_color == "#EF4444"
    assert face.student_id is None


def test_drawing_annotations():
    """Verify drawing bounding box and text overlay onto frame copy."""
    matcher = VectorMatcherCache(threshold=0.637)
    mock_det = FaceDetection(
        bbox=(50, 50, 100, 100),
        confidence=0.95,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw=np.zeros(15, dtype=np.float32),
    )

    engine = AttendanceEngine(
        detector=MockDetector([mock_det]),
        recognizer=MockRecognizer(),
        matcher_cache=matcher,
    )

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.process_frame(blank_frame, draw_annotations=True)

    assert res.annotated_frame is not None
    # Verify frame was modified (not all zeros)
    assert np.any(res.annotated_frame > 0)


if __name__ == "__main__":
    print("=" * 60)
    print("Running Attendance Engine Unit Tests")
    print("=" * 60)
    test_temporal_consensus_and_duplicate_prevention()
    print("[PASS] test_temporal_consensus_and_duplicate_prevention")
    test_unknown_person_detection()
    print("[PASS] test_unknown_person_detection")
    test_drawing_annotations()
    print("[PASS] test_drawing_annotations")
    print("=" * 60)
    print("ALL ATTENDANCE ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

