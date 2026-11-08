"""FaceAttend AI - Demonstration Data Seeder Script.

Populates the database with realistic sample students, attendance sessions,
biometric embeddings, and attendance records so dashboards and charts
are immediately populated for academic evaluation and demonstration.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.matcher import GLOBAL_MATCHER_CACHE
from database import db

SAMPLE_STUDENTS = [
    ("CS-2024-001", "Alice", "Johnson", "COMPUTER SCIENCE", "alice.j@university.edu"),
    ("CS-2024-002", "Bob", "Smith", "COMPUTER SCIENCE", "bob.s@university.edu"),
    ("AI-2024-010", "Charlie", "Brown", "ARTIFICIAL INTELLIGENCE", "charlie.b@university.edu"),
    ("AI-2024-011", "David", "Miller", "ARTIFICIAL INTELLIGENCE", "david.m@university.edu"),
    ("IT-2024-020", "Eva", "Green", "INFORMATION TECHNOLOGY", "eva.g@university.edu"),
    ("IT-2024-021", "Frank", "White", "INFORMATION TECHNOLOGY", "frank.w@university.edu"),
    ("EC-2024-030", "Grace", "Hopper", "ELECTRONICS", "grace.h@university.edu"),
    ("EC-2024-031", "Katherine", "Johnson", "ELECTRONICS", "katherine.j@university.edu"),
    ("CS-2024-003", "Alan", "Turing", "COMPUTER SCIENCE", "alan.t@university.edu"),
    ("AI-2024-012", "Ada", "Lovelace", "ARTIFICIAL INTELLIGENCE", "ada.l@university.edu"),
]

SAMPLE_COURSES = [
    ("CS501", "Computer Vision & Biometrics", "Dr. A. Raman"),
    ("AI602", "Deep Learning & Neural Networks", "Prof. S. Gupta"),
    ("IT403", "Cloud Security & Scalable Systems", "Dr. M. Chen"),
]


def seed_database():
    print("=" * 60)
    print("FaceAttend AI - Seeding Demonstration Data")
    print("=" * 60)

    # 1. Initialize tables
    db.init_db()

    # 2. Add Students and Synthetic Normalized Biometric Embeddings
    print("[*] Enrolling 10 demonstration students with biometric vectors...")
    student_ids = []
    for roll, first, last, dept, email in SAMPLE_STUDENTS:
        existing = db.get_student_by_roll(roll)
        if existing:
            s_id = existing["student_id"]
        else:
            s_id = db.add_student(
                roll_number=roll,
                first_name=first,
                last_name=last,
                department=dept,
                email=email,
            )
        student_ids.append(s_id)

        # Generate 5 realistic synthetic embeddings with small random perturbation
        base_vector = np.random.randn(128).astype(np.float32)
        base_vector /= np.linalg.norm(base_vector)

        samples = []
        for _ in range(5):
            noise = np.random.randn(128).astype(np.float32) * 0.05
            v = base_vector + noise
            v /= np.linalg.norm(v)
            samples.append(v)

        db.delete_face_embeddings(s_id)
        db.save_face_embeddings(
            student_id=s_id,
            embeddings=samples,
            quality_scores=[0.92, 0.95, 0.97, 0.94, 0.96],
        )

    # 3. Create Past and Active Sessions
    print("[*] Creating academic attendance sessions...")
    now = datetime.now()
    session_ids = []

    # Past Sessions (Past 5 days)
    for i in range(5, 0, -1):
        s_date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        course_code, course_name, instructor = SAMPLE_COURSES[i % len(SAMPLE_COURSES)]
        code = f"{course_code}-PAST-D{i}"

        with db.get_db_cursor() as cursor:
            cursor.execute("SELECT session_id FROM sessions WHERE session_code = ?", (code,))
            row = cursor.fetchone()
            if row:
                sess_id = row["session_id"]
            else:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_code, course_code, course_name, instructor, session_date, start_time, end_time, status)
                    VALUES (?, ?, ?, ?, ?, '09:00:00', '10:00:00', 'COMPLETED')
                    """,
                    (code, course_code, course_name, instructor, s_date),
                )
                sess_id = cursor.lastrowid

        session_ids.append(sess_id)

        # Randomly mark 6 to 9 students present
        present_subset = np.random.choice(student_ids, size=np.random.randint(6, 10), replace=False)
        for s_id in present_subset:
            conf = float(np.random.uniform(0.85, 0.99))
            db.mark_attendance(
                session_id=int(sess_id),
                student_id=int(s_id),
                confidence_score=conf,
                method="AUTO_FACE",
                remarks=f"Confidence: {conf:.1%}",
            )

    # 4. Create Active Session for Today
    today_str = now.strftime("%Y-%m-%d")
    active_code = f"CS501-LIVE-{now.strftime('%Y%m%d')}"
    with db.get_db_cursor() as cursor:
        cursor.execute("SELECT session_id FROM sessions WHERE session_code = ?", (active_code,))
        row = cursor.fetchone()
        if row:
            active_id = row["session_id"]
        else:
            cursor.execute(
                """
                INSERT INTO sessions (session_code, course_code, course_name, instructor, session_date, start_time, status)
                VALUES (?, 'CS501', 'Computer Vision & Biometrics', 'Dr. A. Raman', ?, '10:00:00', 'ACTIVE')
                """,
                (active_code, today_str),
            )
            active_id = cursor.lastrowid

    # Mark 3 students already present in today's active session
    for s_id in student_ids[:3]:
        db.mark_attendance(
            session_id=int(active_id),
            student_id=int(s_id),
            confidence_score=0.96,
            method="AUTO_FACE",
            remarks="Live Demonstration Attendance",
        )

    # 5. Refresh in-memory vector cache
    warmed = GLOBAL_MATCHER_CACHE.warm_from_db()

    print(f"[*] Demonstration database successfully seeded!")
    print(f"    - Registered Students: {len(student_ids)}")
    print(f"    - Total Biometric Embeddings in Cache: {warmed}")
    print(f"    - Active Session ID: {active_id} (CS501 - Computer Vision & Biometrics)")
    print("=" * 60)


if __name__ == "__main__":
    seed_database()
