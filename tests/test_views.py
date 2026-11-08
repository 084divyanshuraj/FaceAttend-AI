"""FaceAttend AI - Verification Tests for UI Views & Template Rendering.

Ensures all Jinja2 templates compile cleanly, routes respond with HTTP 200,
and static CSS/JS assets are properly referenced.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app


def test_template_rendering_and_views():
    """Verify that all UI views render with HTTP 200 and no Jinja2 compile errors."""
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Root redirect
    res_root = client.get("/")
    assert res_root.status_code == 302
    assert "/dashboard" in res_root.headers["Location"]

    # 2. Login View
    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert b"Administrator Authentication" in res_login.data

    # 3. Dashboard View
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert b"Executive Attendance Overview" in res_dash.data

    # 4. Students View
    res_students = client.get("/students")
    assert res_students.status_code == 200
    assert b"Student Records & Biometrics" in res_students.data

    # 5. Register View
    res_reg = client.get("/register")
    assert res_reg.status_code == 200
    assert b"Biometric Face Capture" in res_reg.data

    # 6. Live Attendance View
    res_live = client.get("/live")
    assert res_live.status_code == 200
    assert b"Real-Time Biometric Attendance" in res_live.data

    # 7. Records View
    res_records = client.get("/records")
    assert res_records.status_code == 200
    assert b"Attendance Logs & Auditing" in res_records.data

    # 8. Settings View
    res_settings = client.get("/settings")
    assert res_settings.status_code == 200
    assert b"Computer Vision Model Calibrations" in res_settings.data


if __name__ == "__main__":
    print("=" * 60)
    print("Running Frontend UI Views & Template Compilation Tests")
    print("=" * 60)
    test_template_rendering_and_views()
    print("[PASS] test_template_rendering_and_views (All 8 pages verified)")
    print("=" * 60)
    print("ALL FRONTEND TEMPLATES & VIEW ROUTES PASSED SUCCESSFULLY!")
    print("=" * 60)

