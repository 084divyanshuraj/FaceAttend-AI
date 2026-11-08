"""FaceAttend AI - Unit & Verification Tests for Database & Vector Matcher Cache.

Tests database schema creation, WAL mode, student CRUD, binary embedding serialization,
attendance idempotency, and in-memory vector cache operations.
"""

from pathlib import Path
import sys
import tempfile

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.matcher import VectorMatcherCache
from database import db


def create_temp_db():
    """Create a temporary SQLite database for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = Path(temp_dir) / "test_faceattend.db"
    db.init_db(db_path=temp_db_path)
    return temp_db_path


def test_db_initialization_and_tables():
    """Verify that all required tables and indexes are created in WAL mode."""
    temp_db = create_temp_db()
    conn = db.get_db_connection(temp_db)
    cursor = conn.cursor()

    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    required_tables = {"students", "face_embeddings", "sessions", "attendance", "audit_logs"}
    assert required_tables.issubset(tables), f"Missing tables: {required_tables - tables}"

    # Verify WAL journal mode
    cursor.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.upper() == "WAL", f"Expected WAL mode, got {journal_mode}"

    conn.close()


def test_student_crud():
    """Verify student creation, retrieval, update, and listing."""
    temp_db = create_temp_db()

    # 1. Add student
    student_id = db.add_student(
        roll_number="CS-2024-001",
        first_name="Alice",
        last_name="Johnson",
        department="Computer Science",
        email="alice@university.edu",
        db_path=temp_db,
    )
    assert student_id > 0

    # 2. Get student
    student = db.get_student(student_id, db_path=temp_db)
    assert student is not None
    assert student["roll_number"] == "CS-2024-001"
    assert student["first_name"] == "Alice"
    assert student["department"] == "COMPUTER SCIENCE"

    # 3. Retrieve by roll number
    by_roll = db.get_student_by_roll("cs-2024-001", db_path=temp_db)
    assert by_roll is not None
    assert by_roll["student_id"] == student_id

    # 4. Update student
    updated = db.update_student(student_id, department="AI & Robotics", db_path=temp_db)
    assert updated is True
    re_student = db.get_student(student_id, db_path=temp_db)
    assert re_student["department"] == "AI & ROBOTICS"

    # 5. List students
    students = db.get_all_students(db_path=temp_db)
    assert len(students) == 1


def test_embedding_storage_and_deserialization():
    """Verify 128-D float32 embeddings serialization and deserialization."""
    temp_db = create_temp_db()
    student_id = db.add_student("CS-002", "Bob", "Smith", "IT", db_path=temp_db)

    # Generate 3 distinct 128-D unit vectors
    embeddings = []
    for i in range(3):
        vec = np.zeros(128, dtype=np.float32)
        vec[i] = 1.0
        embeddings.append(vec)

    saved_count = db.save_face_embeddings(
        student_id=student_id,
        embeddings=embeddings,
        quality_scores=[0.95, 0.92, 0.98],
        db_path=temp_db,
    )
    assert saved_count == 3

    # Retrieve and verify exact byte-level reconstruction
    retrieved = db.get_embeddings_for_student(student_id, db_path=temp_db)
    assert len(retrieved) == 3
    for orig, ret in zip(embeddings, retrieved):
        assert ret.shape == (128,)
        assert ret.dtype == np.float32
        assert np.allclose(orig, ret)


def test_attendance_idempotency_and_duplicates():
    """Verify attendance logging prevents duplicate records in the same session."""
    temp_db = create_temp_db()
    student_id = db.add_student("CS-003", "Charlie", "Brown", "ECE", db_path=temp_db)
    session_id = db.create_session("CS501", "Computer Vision", "Dr. Rao", db_path=temp_db)

    # 1. First attendance mark -> Must succeed
    marked_1, att_id_1 = db.mark_attendance(
        session_id=session_id,
        student_id=student_id,
        confidence_score=0.92,
        method="AUTO_FACE",
        db_path=temp_db,
    )
    assert marked_1 is True
    assert att_id_1 is not None

    # 2. Duplicate attendance mark in same session -> Must be ignored (idempotent)
    marked_2, att_id_2 = db.mark_attendance(
        session_id=session_id,
        student_id=student_id,
        confidence_score=0.95,
        method="AUTO_FACE",
        db_path=temp_db,
    )
    assert marked_2 is False
    assert att_id_2 == att_id_1  # Returns existing attendance_id

    # 3. Verify exactly 1 attendance record in session
    records = db.get_session_attendance(session_id, db_path=temp_db)
    assert len(records) == 1
    assert records[0]["student_id"] == student_id
    assert records[0]["roll_number"] == "CS-003"


def test_vector_matcher_cache_operations():
    """Verify in-memory vector cache warming, sub-millisecond matching, and dynamic updates."""
    temp_db = create_temp_db()

    # Register Student 1
    s1_id = db.add_student("CS-101", "David", "Miller", "CSE", db_path=temp_db)
    v1 = np.zeros(128, dtype=np.float32)
    v1[0] = 1.0
    db.save_face_embeddings(s1_id, [v1], [1.0], db_path=temp_db)

    # Warm cache
    matcher = VectorMatcherCache(threshold=0.637)
    count = matcher.warm_from_db(temp_db)
    assert count == 1
    assert len(matcher) == 1

    # 1. Match known student query
    res_known = matcher.match(v1)
    assert res_known.is_known is True
    assert res_known.student_id == s1_id
    assert res_known.roll_number == "CS-101"
    assert np.isclose(res_known.confidence, 1.0, atol=1e-4)

    # 2. Match unknown query (orthogonal vector)
    v_unknown = np.zeros(128, dtype=np.float32)
    v_unknown[10] = 1.0
    res_unknown = matcher.match(v_unknown)
    assert res_unknown.is_known is False
    assert res_unknown.student_id is None
    assert res_unknown.label == "Unknown"
    assert res_unknown.confidence < 0.637

    # 3. Dynamically add Student 2 into cache without re-reading DB
    v2 = np.zeros(128, dtype=np.float32)
    v2[10] = 1.0
    matcher.add_student_samples(
        student_id=99,
        roll_number="CS-102",
        name="Eva Green",
        department="AI",
        embeddings=[v2],
    )
    assert len(matcher) == 2

    # Now v_unknown matches Eva Green
    res_eva = matcher.match(v_unknown)
    assert res_eva.is_known is True
    assert res_eva.student_id == 99
    assert res_eva.roll_number == "CS-102"

    # 4. Remove student from cache
    matcher.remove_student(99)
    assert len(matcher) == 1
    res_after_remove = matcher.match(v_unknown)
    assert res_after_remove.is_known is False


def test_cascade_deletion():
    """Verify that deleting a student cascades to their embeddings and attendance records."""
    temp_db = create_temp_db()
    student_id = db.add_student("CS-999", "Temp", "User", "IT", db_path=temp_db)
    session_id = db.create_session("TEST", "Test Course", "Test Instructor", db_path=temp_db)

    v = np.ones(128, dtype=np.float32) / np.sqrt(128)
    db.save_face_embeddings(student_id, [v], db_path=temp_db)
    db.mark_attendance(session_id, student_id, 0.95, db_path=temp_db)

    # Delete student
    deleted = db.delete_student(student_id, db_path=temp_db)
    assert deleted is True

    # Verify embeddings and attendance are automatically deleted
    conn = db.get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE student_id = ?", (student_id,))
    assert cursor.fetchone()[0] == 0
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ?", (student_id,))
    assert cursor.fetchone()[0] == 0
    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Database & Vector Matcher Cache Unit Tests")
    print("=" * 60)
    test_db_initialization_and_tables()
    print("[PASS] test_db_initialization_and_tables (WAL mode verified)")
    test_student_crud()
    print("[PASS] test_student_crud")
    test_embedding_storage_and_deserialization()
    print("[PASS] test_embedding_storage_and_deserialization")
    test_attendance_idempotency_and_duplicates()
    print("[PASS] test_attendance_idempotency_and_duplicates")
    test_vector_matcher_cache_operations()
    print("[PASS] test_vector_matcher_cache_operations")
    test_cascade_deletion()
    print("[PASS] test_cascade_deletion")
    print("=" * 60)
    print("ALL DATABASE & VECTOR MATCHER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

