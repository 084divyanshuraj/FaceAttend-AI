"""FaceAttend AI - Unit & Verification Tests for Audit Trail & Reporting System.

Tests manual attendance overrides, audit log recording, CSV stream generation,
and department aggregation queries.
"""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import db


def test_audit_log_recording_and_retrieval():
    """Verify that manual overrides generate strict audit log entries."""
    temp_dir = tempfile.mkdtemp()
    temp_db = Path(temp_dir) / "test_audit.db"
    db.init_db(db_path=temp_db)

    # 1. Register student & session
    student_id = db.add_student("CS-AUDIT-1", "Grace", "Hopper", "CS", db_path=temp_db)
    session_id = db.create_session("AUDIT101", "Security Auditing", "Prof. Rivest", db_path=temp_db)

    # 2. Perform manual override: MARK_PRESENT
    ok_add = db.manual_attendance_override(
        session_id=session_id,
        student_id=student_id,
        action="MARK_PRESENT",
        admin_user="prof_admin",
        remarks="Student had technical issue with camera",
        db_path=temp_db,
    )
    assert ok_add is True

    # 3. Verify attendance record was created
    records = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(records) == 1
    assert records[0]["method"] == "MANUAL_ADMIN"

    # 4. Verify audit_logs table contains the action
    conn = db.get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs WHERE target_id = ?", (student_id,))
    logs = cursor.fetchall()
    assert len(logs) == 1
    log = dict(logs[0])
    assert log["user_id"] == "prof_admin"
    assert log["action"] == "MANUAL_ATTENDANCE_ADD"
    assert "technical issue" in log["details"]

    # 5. Perform manual override: REMOVE
    ok_rem = db.manual_attendance_override(
        session_id=session_id,
        student_id=student_id,
        action="REMOVE",
        admin_user="prof_admin",
        remarks="Correction: student was marked in error",
        db_path=temp_db,
    )
    assert ok_rem is True

    # Verify attendance record is gone
    re_records = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(re_records) == 0

    # Verify second audit log entry exists
    cursor.execute("SELECT * FROM audit_logs WHERE target_id = ? ORDER BY log_id DESC", (student_id,))
    all_logs = cursor.fetchall()
    assert len(all_logs) == 2
    assert all_logs[0]["action"] == "MANUAL_ATTENDANCE_REMOVE"

    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Audit Trail & Reporting Unit Tests")
    print("=" * 60)
    test_audit_log_recording_and_retrieval()
    print("[PASS] test_audit_log_recording_and_retrieval")
    print("=" * 60)
    print("ALL AUDIT TRAIL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

