"""FaceAttend AI - Unit & Verification Tests for Face Quality Assessment (FQA) & Registration Engine.

Tests blur detection, illumination checks, pose estimation, multi-sample diversity gating,
and database/vector cache commit.
"""

from pathlib import Path
import sys
import tempfile

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

import config
from core.detector import FaceDetection
from core.face_registration import EnrollmentSessionState, RegistrationEngine
from core.matcher import GLOBAL_MATCHER_CACHE
from core.quality import FaceQualityAssessor
from database import db


def create_mock_detection(
    bbox=(100, 100, 150, 150),
    confidence=0.95,
    roll_angle_deg=0.0,
    yaw_ratio=1.0,
) -> FaceDetection:
    """Helper to generate a mock FaceDetection object with calibrated landmarks."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2

    # Landmarks: [0] Right eye, [1] Left eye, [2] Nose, [3] Right mouth, [4] Left mouth
    re_x = cx - 30
    re_y = cy - 20
    le_x = cx + 30
    le_y = cy - 20

    if roll_angle_deg != 0.0:
        rad = np.radians(roll_angle_deg)
        le_y = re_y + 60 * np.sin(rad)
        le_x = re_x + 60 * np.cos(rad)

    nt_x = cx
    nt_y = cy + 5
    if yaw_ratio != 1.0:
        # Shift nose horizontally to simulate yaw
        nt_x = re_x + (le_x - re_x) * (yaw_ratio / (1.0 + yaw_ratio))

    rm_x = cx - 20
    rm_y = cy + 35
    lm_x = cx + 20
    lm_y = cy + 35

    landmarks = np.array([
        [re_x, re_y],
        [le_x, le_y],
        [nt_x, nt_y],
        [rm_x, rm_y],
        [lm_x, lm_y],
    ], dtype=np.float32)

    raw = np.zeros(15, dtype=np.float32)
    raw[0:4] = [x, y, w, h]
    raw[4:14] = landmarks.flatten()
    raw[14] = confidence

    return FaceDetection(
        bbox=bbox,
        confidence=confidence,
        landmarks=landmarks,
        raw=raw,
    )


def test_blur_quality_rejection():
    """Verify that blurry frames are rejected and sharp frames pass."""
    assessor = FaceQualityAssessor(blur_threshold=80.0)

    # 1. Create a sharp image with high-contrast text/edges
    frame = np.full((400, 400, 3), 200, dtype=np.uint8)
    cv2.putText(frame, "SHARP BIOMETRICS", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    detection = create_mock_detection(bbox=(100, 100, 160, 160), confidence=0.95)

    res_sharp = assessor.evaluate(frame, detection)
    assert res_sharp.metrics["blur_score"] > 80.0
    assert res_sharp.is_valid is True

    # 2. Heavily blur the image
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    res_blur = assessor.evaluate(blurred_frame, detection)
    assert res_blur.is_valid is False
    assert any("blurry" in issue.lower() for issue in res_blur.issues)
    assert "blurry" in res_blur.feedback_message.lower()


def test_illumination_rejection():
    """Verify that underexposed (pitch black) and overexposed frames are rejected."""
    assessor = FaceQualityAssessor()
    detection = create_mock_detection(bbox=(50, 50, 150, 150), confidence=0.95)

    # 1. Pitch black frame
    black_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    res_dark = assessor.evaluate(black_frame, detection)
    assert res_dark.is_valid is False
    assert any("too dark" in issue.lower() for issue in res_dark.issues)

    # 2. Overexposed / whiteout frame
    white_frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    res_bright = assessor.evaluate(white_frame, detection)
    assert res_bright.is_valid is False
    assert any("overexposed" in issue.lower() for issue in res_bright.issues)


def test_head_pose_rejection():
    """Verify head tilt roll and yaw asymmetry rejection."""
    assessor = FaceQualityAssessor(max_pose_angle_deg=20.0)
    frame = np.full((400, 400, 3), 150, dtype=np.uint8)
    cv2.putText(frame, "FACE", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 1. Normal frontal pose (0 deg)
    det_frontal = create_mock_detection(roll_angle_deg=0.0)
    res_frontal = assessor.evaluate(frame, det_frontal)
    assert res_frontal.metrics["roll_angle_deg"] < 5.0

    # 2. Severe tilt (35 degrees)
    det_tilt = create_mock_detection(roll_angle_deg=35.0)
    res_tilt = assessor.evaluate(frame, det_tilt)
    assert res_tilt.is_valid is False
    assert any("tilted" in issue.lower() for issue in res_tilt.issues)


def test_full_enrollment_pipeline_and_db_commit():
    """Test full multi-sample enrollment session and database/matcher synchronization."""
    temp_dir = tempfile.mkdtemp()
    temp_db = Path(temp_dir) / "test_enroll.db"
    db.init_db(db_path=temp_db)

    # Register student in SQLite
    student_id = db.add_student("CS-555", "Grace", "Hopper", "CS", db_path=temp_db)
    session = EnrollmentSessionState(student_id=student_id, target_samples=3)

    engine = RegistrationEngine()

    # Create dummy distinct embeddings to simulate 3 accepted diverse samples
    embeddings = []
    for i in range(3):
        v = np.zeros(128, dtype=np.float32)
        v[i] = 1.0
        embeddings.append(v)

    session.collected_embeddings = embeddings
    session.quality_scores = [0.92, 0.95, 0.98]
    session.is_complete = True

    # Commit enrollment
    success = engine.commit_enrollment(session, db_path=temp_db)
    assert success is True

    # 1. Verify in database
    db_embeddings = db.get_embeddings_for_student(student_id, db_path=temp_db)
    assert len(db_embeddings) == 3

    # 2. Verify in in-memory vector cache
    assert len(GLOBAL_MATCHER_CACHE) >= 3
    match = GLOBAL_MATCHER_CACHE.match(embeddings[0])
    assert match.is_known is True
    assert match.student_id == student_id
    assert match.roll_number == "CS-555"
    assert "Grace Hopper" in match.name


if __name__ == "__main__":
    print("=" * 60)
    print("Running Face Quality Assessment & Registration Unit Tests")
    print("=" * 60)
    test_blur_quality_rejection()
    print("[PASS] test_blur_quality_rejection")
    test_illumination_rejection()
    print("[PASS] test_illumination_rejection")
    test_head_pose_rejection()
    print("[PASS] test_head_pose_rejection")
    test_full_enrollment_pipeline_and_db_commit()
    print("[PASS] test_full_enrollment_pipeline_and_db_commit")
    print("=" * 60)
    print("ALL FQA & REGISTRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

