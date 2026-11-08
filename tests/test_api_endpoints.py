"""FaceAttend AI - Unit & Integration Tests for Flask REST APIs.

Tests authentication, student CRUD, session management, live frame processing,
and CSV report export endpoints.
"""

import base64
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app import create_app
from database import db


def setup_test_app():
    """Create Flask test client configured for isolated testing."""
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint():
    """Verify system healthcheck route."""
    client = setup_test_app()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "enrolled_biometrics" in data


def test_auth_endpoints():
    """Verify administrator login, status, and logout."""
    client = setup_test_app()

    # 1. Invalid credentials
    res_bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert res_bad.status_code == 401
    assert res_bad.get_json()["success"] is False

    # 2. Valid credentials
    res_good = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res_good.status_code == 200
    assert res_good.get_json()["success"] is True

    # 3. Status check
    res_status = client.get("/api/auth/status")
    assert res_status.status_code == 200
    assert res_status.get_json()["authenticated"] is True

    # 4. Logout
    res_logout = client.post("/api/auth/logout")
    assert res_logout.status_code == 200
    assert res_logout.get_json()["success"] is True


def test_student_endpoints():
    """Verify student registration, retrieval, and listing."""
    client = setup_test_app()

    roll = f"TEST-API-{np.random.randint(1000, 9999)}"

    # 1. Register student
    res_post = client.post("/api/students", json={
        "roll_number": roll,
        "first_name": "Katherine",
        "last_name": "Johnson",
        "department": "AEROSPACE",
        "email": f"{roll.lower()}@university.edu",
    })
    assert res_post.status_code == 201
    student_id = res_post.get_json()["student_id"]
    assert student_id > 0

    # 2. Duplicate registration should return 409
    res_dup = client.post("/api/students", json={
        "roll_number": roll,
        "first_name": "Duplicate",
        "last_name": "User",
        "department": "MATH",
    })
    assert res_dup.status_code == 409

    # 3. Retrieve student profile
    res_get = client.get(f"/api/students/{student_id}")
    assert res_get.status_code == 200
    assert res_get.get_json()["student"]["roll_number"] == roll

    # 4. List all students
    res_list = client.get("/api/students")
    assert res_list.status_code == 200
    students = res_list.get_json()["students"]
    assert any(s["student_id"] == student_id for s in students)


def test_session_lifecycle_endpoints():
    """Verify attendance session creation, active retrieval, and closure."""
    client = setup_test_app()

    # 1. Start new session
    res_start = client.post("/api/sessions", json={
        "course_code": "CV401",
        "course_name": "Computer Vision & Biometrics",
        "instructor": "Dr. Raman",
    })
    assert res_start.status_code == 201
    session_id = res_start.get_json()["session"]["session_id"]
    assert session_id > 0

    # 2. Get active session
    res_active = client.get("/api/sessions/active")
    assert res_active.status_code == 200
    active = res_active.get_json()["active_session"]
    assert active is not None
    assert active["session_id"] == session_id

    # 3. Close session
    res_close = client.post(f"/api/sessions/{session_id}/close")
    assert res_close.status_code == 200
    assert res_close.get_json()["success"] is True


def test_live_frame_processing_endpoint():
    """Verify /api/attendance/process-frame with base64 encoded frame."""
    client = setup_test_app()

    # Generate a dummy blank image encoded as base64 JPEG
    blank_img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", blank_img)
    base64_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    res = client.post("/api/attendance/process-frame", json={
        "image": base64_str,
        "session_id": None,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "faces_detected" in data
    assert data["faces_detected"] == 0
    assert "session_stats" in data


def test_analytics_and_csv_export():
    """Verify dashboard stats and CSV streaming export."""
    client = setup_test_app()

    # 1. Dashboard summary
    res_dash = client.get("/api/analytics/dashboard")
    assert res_dash.status_code == 200
    stats = res_dash.get_json()["stats"]
    assert "total_students" in stats
    assert "total_sessions" in stats

    # 2. Daily trend
    res_trend = client.get("/api/analytics/daily-trend")
    assert res_trend.status_code == 200
    assert "trend" in res_trend.get_json()

    # 3. CSV report export
    res_csv = client.get("/api/reports/export-csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["Content-Type"]
    assert "attachment; filename=" in res_csv.headers["Content-Disposition"]
    assert "Roll Number" in res_csv.data.decode("utf-8")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Flask REST API Unit & Integration Tests")
    print("=" * 60)
    test_health_endpoint()
    print("[PASS] test_health_endpoint")
    test_auth_endpoints()
    print("[PASS] test_auth_endpoints")
    test_student_endpoints()
    print("[PASS] test_student_endpoints")
    test_session_lifecycle_endpoints()
    print("[PASS] test_session_lifecycle_endpoints")
    test_live_frame_processing_endpoint()
    print("[PASS] test_live_frame_processing_endpoint")
    test_analytics_and_csv_export()
    print("[PASS] test_analytics_and_csv_export")
    print("=" * 60)
    print("ALL FLASK REST API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

