"""FaceAttend AI - Student Management & Biometric Enrollment Blueprint.

Handles student registration, updates, profile queries, biometric deletion,
and multi-sample live camera enrollment.
"""

import base64
from typing import Dict, Optional
import cv2
from flask import Blueprint, jsonify, request
import numpy as np

import config
from core.face_registration import EnrollmentSessionState, RegistrationEngine
from core.matcher import GLOBAL_MATCHER_CACHE
from database import db

students_bp = Blueprint("students", __name__)

# Active enrollment sessions buffer: student_id -> EnrollmentSessionState
ACTIVE_ENROLLMENT_SESSIONS: Dict[int, EnrollmentSessionState] = {}
REGISTRATION_ENGINE = RegistrationEngine()


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 data URI or raw string to OpenCV BGR image."""
    if not base64_str:
        return None
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@students_bp.route("/api/students", methods=["GET"])
def list_students():
    """Retrieve all students with enrolled biometric samples count."""
    students = db.get_all_students()
    result = []
    for s in students:
        embeddings = db.get_embeddings_for_student(s["student_id"])
        s_data = dict(s)
        s_data["enrolled_samples"] = len(embeddings)
        s_data["is_enrolled"] = len(embeddings) >= config.QUALITY_CONFIG["samples_per_student"]
        result.append(s_data)

    return jsonify({"success": True, "students": result})


@students_bp.route("/api/students", methods=["POST"])
def register_student():
    """Register a new student record."""
    data = request.get_json(silent=True) or {}
    roll_number = data.get("roll_number", "").strip()
    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()
    department = data.get("department", "").strip()
    email = data.get("email", "").strip() or None

    if not (roll_number and first_name and last_name and department):
        return jsonify({
            "success": False,
            "error": "Roll number, first name, last name, and department are required.",
        }), 400

    existing = db.get_student_by_roll(roll_number)
    if existing:
        return jsonify({
            "success": False,
            "error": f"Student with roll number '{roll_number}' already exists.",
        }), 409

    try:
        student_id = db.add_student(
            roll_number=roll_number,
            first_name=first_name,
            last_name=last_name,
            department=department,
            email=email,
        )
        return jsonify({
            "success": True,
            "message": "Student registered successfully.",
            "student_id": student_id,
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@students_bp.route("/api/students/<int:student_id>", methods=["GET"])
def get_student_profile(student_id: int):
    """Retrieve detailed student profile."""
    student = db.get_student(student_id)
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404

    embeddings = db.get_embeddings_for_student(student_id)
    profile = dict(student)
    profile["enrolled_samples"] = len(embeddings)
    profile["is_enrolled"] = len(embeddings) >= config.QUALITY_CONFIG["samples_per_student"]

    return jsonify({"success": True, "student": profile})


@students_bp.route("/api/students/<int:student_id>", methods=["PUT"])
def update_student_record(student_id: int):
    """Update student information."""
    data = request.get_json(silent=True) or {}
    success = db.update_student(
        student_id=student_id,
        first_name=data.get("first_name"),
        last_name=data.get("last_name"),
        department=data.get("department"),
        email=data.get("email"),
        is_active=data.get("is_active"),
    )
    if not success:
        return jsonify({"success": False, "error": "Failed to update student record."}), 400

    return jsonify({"success": True, "message": "Student record updated."})


@students_bp.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student_record(student_id: int):
    """Delete student and all associated biometric embeddings (CASCADE)."""
    student = db.get_student(student_id)
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404

    # Remove from in-memory cache
    GLOBAL_MATCHER_CACHE.remove_student(student_id)

    # Delete from database
    success = db.delete_student(student_id)
    if not success:
        return jsonify({"success": False, "error": "Failed to delete student."}), 500

    # Clean active enrollment session if any
    ACTIVE_ENROLLMENT_SESSIONS.pop(student_id, None)

    return jsonify({
        "success": True,
        "message": f"Student '{student['roll_number']}' and all biometric data deleted successfully.",
    })


# ============================================================================
# Live Biometric Enrollment Endpoints
# ============================================================================

@students_bp.route("/api/students/<int:student_id>/enroll-frame", methods=["POST"])
def process_enrollment_frame(student_id: int):
    """Process a live camera frame to capture biometric face samples."""
    student = db.get_student(student_id)
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404

    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"success": False, "error": "No image data provided."}), 400

    frame = decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"success": False, "error": "Invalid image format."}), 400

    # Retrieve or initialize session state
    if student_id not in ACTIVE_ENROLLMENT_SESSIONS:
        ACTIVE_ENROLLMENT_SESSIONS[student_id] = EnrollmentSessionState(
            student_id=student_id,
            target_samples=config.QUALITY_CONFIG["samples_per_student"],
        )

    session_state = ACTIVE_ENROLLMENT_SESSIONS[student_id]

    # Process frame through registration engine
    result = REGISTRATION_ENGINE.process_frame(frame, session_state)

    auto_committed = False
    if session_state.is_complete:
        # Clear existing embeddings before committing new set
        db.delete_face_embeddings(student_id)
        GLOBAL_MATCHER_CACHE.remove_student(student_id)

        committed = REGISTRATION_ENGINE.commit_enrollment(session_state)
        auto_committed = committed
        ACTIVE_ENROLLMENT_SESSIONS.pop(student_id, None)

    return jsonify({
        "success": True,
        "status": result.status,
        "samples_collected": result.samples_collected,
        "samples_required": result.samples_required,
        "message": result.message,
        "is_complete": session_state.is_complete,
        "auto_committed": auto_committed,
        "quality_score": result.quality_result.quality_score if result.quality_result else None,
        "issues": result.quality_result.issues if result.quality_result else [],
    })


@students_bp.route("/api/students/<int:student_id>/reset-enrollment", methods=["POST"])
def reset_enrollment(student_id: int):
    """Reset the active multi-sample enrollment buffer."""
    ACTIVE_ENROLLMENT_SESSIONS.pop(student_id, None)
    return jsonify({"success": True, "message": "Enrollment session reset."})

