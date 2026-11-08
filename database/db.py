"""FaceAttend AI - SQLite Database Access & ORM-Free Data Layer.

Enforces SQLite WAL mode, foreign keys, thread-safe transactions,
and binary serialization for 128-dimensional facial embeddings.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_DB_PATH = config.DATABASE_PATH


def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Create a configured SQLite database connection.

    Enforces WAL journaling and foreign keys.
    """
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_cursor(db_path: Optional[Union[str, Path]] = None):
    """Context manager for safe database transactions with auto-commit/rollback."""
    conn = get_db_connection(db_path)
    try:
        yield conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(
    schema_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Initialize database tables and indexes from schema.sql."""
    schema_file = Path(schema_path or SCHEMA_PATH)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found at '{schema_file}'.")

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = get_db_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# Student Operations
# ============================================================================

def add_student(
    roll_number: str,
    first_name: str,
    last_name: str,
    department: str,
    email: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert a new student and return the new student_id."""
    sql = """
        INSERT INTO students (roll_number, first_name, last_name, department, email)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(
            sql,
            (
                roll_number.strip().upper(),
                first_name.strip().title(),
                last_name.strip().title(),
                department.strip().upper(),
                email.strip().lower() if email else None,
            ),
        )
        return cursor.lastrowid


def get_student(
    student_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve student record by student_id."""
    sql = "SELECT * FROM students WHERE student_id = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_student_by_roll(
    roll_number: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve student record by roll number."""
    sql = "SELECT * FROM students WHERE roll_number = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (roll_number.strip().upper(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_students(
    active_only: bool = False,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all students, optionally filtering by active status."""
    if active_only:
        sql = "SELECT * FROM students WHERE is_active = 1 ORDER BY roll_number ASC"
    else:
        sql = "SELECT * FROM students ORDER BY roll_number ASC"

    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_student(
    student_id: int,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    department: Optional[str] = None,
    email: Optional[str] = None,
    is_active: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Update fields on an existing student record."""
    fields = []
    values = []

    if first_name is not None:
        fields.append("first_name = ?")
        values.append(first_name.strip().title())
    if last_name is not None:
        fields.append("last_name = ?")
        values.append(last_name.strip().title())
    if department is not None:
        fields.append("department = ?")
        values.append(department.strip().upper())
    if email is not None:
        fields.append("email = ?")
        values.append(email.strip().lower())
    if is_active is not None:
        fields.append("is_active = ?")
        values.append(1 if is_active else 0)

    if not fields:
        return False

    values.append(student_id)
    sql = f"UPDATE students SET {', '.join(fields)} WHERE student_id = ?"

    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, tuple(values))
        return cursor.rowcount > 0


def delete_student(
    student_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete student and all associated embeddings and attendance (CASCADE)."""
    sql = "DELETE FROM students WHERE student_id = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (student_id,))
        return cursor.rowcount > 0


# ============================================================================
# Face Embedding Operations (Binary BLOB Serialization)
# ============================================================================

def save_face_embeddings(
    student_id: int,
    embeddings: List[np.ndarray],
    quality_scores: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Save a list of 128-D float32 embeddings for a student as binary blobs.

    Args:
        student_id: ID of the registered student.
        embeddings: List of numpy arrays, each shape (128,).
        quality_scores: Optional list of quality scores per sample.

    Returns:
        Number of saved embeddings.
    """
    if not embeddings:
        return 0

    scores = quality_scores or [1.0] * len(embeddings)
    sql = """
        INSERT INTO face_embeddings (student_id, embedding_blob, sample_index, quality_score)
        VALUES (?, ?, ?, ?)
    """
    rows = []
    for idx, (emb, score) in enumerate(zip(embeddings, scores)):
        vec = emb.flatten().astype(np.float32)
        if vec.shape[0] != 128:
            raise ValueError(f"Embedding must be 128-dimensional, got {vec.shape[0]}")
        blob = vec.tobytes()
        rows.append((student_id, blob, idx, float(score)))

    with get_db_cursor(db_path) as cursor:
        cursor.executemany(sql, rows)
        return len(rows)


def get_embeddings_for_student(
    student_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> List[np.ndarray]:
    """Retrieve all 128-D embeddings for a specific student."""
    sql = """
        SELECT embedding_blob FROM face_embeddings
        WHERE student_id = ? ORDER BY sample_index ASC
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (student_id,))
        rows = cursor.fetchall()
        embeddings = []
        for r in rows:
            vec = np.frombuffer(r["embedding_blob"], dtype=np.float32)
            embeddings.append(vec)
        return embeddings


def get_all_embeddings_with_metadata(
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all active student embeddings with metadata for cache warming.

    Returns:
        List of dicts: {'student_id', 'roll_number', 'name', 'department', 'embedding'}
    """
    sql = """
        SELECT
            s.student_id,
            s.roll_number,
            s.first_name || ' ' || s.last_name AS name,
            s.department,
            fe.embedding_blob
        FROM face_embeddings fe
        JOIN students s ON fe.student_id = s.student_id
        WHERE s.is_active = 1
        ORDER BY s.student_id ASC, fe.sample_index ASC
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        records = []
        for r in rows:
            vec = np.frombuffer(r["embedding_blob"], dtype=np.float32)
            records.append({
                "student_id": r["student_id"],
                "roll_number": r["roll_number"],
                "name": r["name"],
                "department": r["department"],
                "embedding": vec,
            })
        return records


def delete_face_embeddings(
    student_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Delete all stored facial embeddings for a student."""
    sql = "DELETE FROM face_embeddings WHERE student_id = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (student_id,))
        return cursor.rowcount


# ============================================================================
# Attendance Session Operations
# ============================================================================

def create_session(
    course_code: str,
    course_name: str,
    instructor: str,
    session_date: Optional[str] = None,
    start_time: Optional[str] = None,
    session_code: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Create a new attendance session."""
    now = datetime.now()
    s_date = session_date or now.strftime("%Y-%m-%d")
    s_time = start_time or now.strftime("%H:%M:%S")
    code = session_code or f"{course_code.upper()}-{now.strftime('%Y%m%d%H%M%S')}"

    sql = """
        INSERT INTO sessions (session_code, course_code, course_name, instructor, session_date, start_time, status)
        VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(
            sql,
            (
                code,
                course_code.strip().upper(),
                course_name.strip(),
                instructor.strip(),
                s_date,
                s_time,
            ),
        )
        return cursor.lastrowid


def get_session(
    session_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve session details by session_id."""
    sql = "SELECT * FROM sessions WHERE session_id = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_active_session(
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve the latest active session, if any."""
    sql = "SELECT * FROM sessions WHERE status = 'ACTIVE' ORDER BY session_id DESC LIMIT 1"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql)
        row = cursor.fetchone()
        return dict(row) if row else None


def close_session(
    session_id: int,
    end_time: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Mark a session as COMPLETED."""
    e_time = end_time or datetime.now().strftime("%H:%M:%S")
    sql = "UPDATE sessions SET status = 'COMPLETED', end_time = ? WHERE session_id = ?"
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (e_time, session_id))
        return cursor.rowcount > 0


# ============================================================================
# Attendance Logging & Idempotency
# ============================================================================

def mark_attendance(
    session_id: int,
    student_id: int,
    confidence_score: float,
    method: str = "AUTO_FACE",
    remarks: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Optional[int]]:
    """Mark attendance for a student in a session.

    Guarantees idempotency via UNIQUE(session_id, student_id).
    Returns:
        (marked: bool, attendance_id: Optional[int])
        marked is True if a new record was inserted; False if student was already marked present.
    """
    sql = """
        INSERT OR IGNORE INTO attendance (session_id, student_id, confidence_score, method, remarks)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (session_id, student_id, float(confidence_score), method, remarks))
        if cursor.rowcount > 0:
            return True, cursor.lastrowid

        # Already marked present
        cursor.execute(
            "SELECT attendance_id FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        )
        row = cursor.fetchone()
        return False, (row["attendance_id"] if row else None)


def get_session_attendance(
    session_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve full attendance list for a specific session with student details."""
    sql = """
        SELECT
            a.attendance_id,
            a.session_id,
            a.student_id,
            s.roll_number,
            s.first_name || ' ' || s.last_name AS name,
            s.department,
            a.marked_at,
            a.confidence_score,
            a.method,
            a.remarks
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE a.session_id = ?
        ORDER BY a.marked_at DESC
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute(sql, (session_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def manual_attendance_override(
    session_id: int,
    student_id: int,
    action: str,  # "MARK_PRESENT" or "REMOVE"
    admin_user: str = "admin",
    remarks: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Perform an administrative attendance override with audit trail."""
    action = action.upper()
    with get_db_cursor(db_path) as cursor:
        if action == "MARK_PRESENT":
            cursor.execute(
                """
                INSERT OR REPLACE INTO attendance (session_id, student_id, confidence_score, method, remarks)
                VALUES (?, ?, 1.0, 'MANUAL_ADMIN', ?)
                """,
                (session_id, student_id, remarks or "Manual admin override"),
            )
            audit_action = "MANUAL_ATTENDANCE_ADD"
        elif action == "REMOVE":
            cursor.execute(
                "DELETE FROM attendance WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            )
            audit_action = "MANUAL_ATTENDANCE_REMOVE"
        else:
            raise ValueError(f"Unknown override action: '{action}'")

        # Log audit trail
        cursor.execute(
            """
            INSERT INTO audit_logs (user_id, action, target_entity, target_id, details)
            VALUES (?, ?, 'attendance', ?, ?)
            """,
            (
                admin_user,
                audit_action,
                student_id,
                f"Session {session_id} - {remarks or 'No remarks'}",
            ),
        )
        return True


# ============================================================================
# Analytics & Reporting
# ============================================================================

def get_dashboard_stats(
    session_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Calculate live high-level dashboard metrics."""
    with get_db_cursor(db_path) as cursor:
        # Total registered active students
        cursor.execute("SELECT COUNT(*) AS total FROM students WHERE is_active = 1")
        total_students = cursor.fetchone()["total"]

        # Present count in active session
        if session_id:
            cursor.execute(
                "SELECT COUNT(*) AS present FROM attendance WHERE session_id = ?",
                (session_id,),
            )
            present_students = cursor.fetchone()["present"]
        else:
            present_students = 0

        absent_students = max(0, total_students - present_students)
        pct = (present_students / total_students * 100.0) if total_students > 0 else 0.0

        return {
            "total_students": total_students,
            "present_students": present_students,
            "absent_students": absent_students,
            "attendance_percentage": round(pct, 1),
        }

