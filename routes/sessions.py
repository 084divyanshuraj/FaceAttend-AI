"""FaceAttend AI - Attendance Sessions Management Blueprint.

Handles creation, retrieval, active session tracking, and session completion.
"""

from flask import Blueprint, jsonify, request
from database import db

sessions_bp = Blueprint("sessions", __name__)


@sessions_bp.route("/api/sessions", methods=["GET"])
def list_sessions():
    """Retrieve all attendance sessions with attendance counts."""
    with db.get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                s.*,
                COUNT(a.attendance_id) AS total_present
            FROM sessions s
            LEFT JOIN attendance a ON s.session_id = a.session_id
            GROUP BY s.session_id
            ORDER BY s.session_id DESC
            """
        )
        rows = cursor.fetchall()
        return jsonify({"success": True, "sessions": [dict(r) for r in rows]})


@sessions_bp.route("/api/sessions", methods=["POST"])
def start_session():
    """Create and start a new attendance session."""
    data = request.get_json(silent=True) or {}
    course_code = data.get("course_code", "").strip()
    course_name = data.get("course_name", "").strip()
    instructor = data.get("instructor", "").strip()
    session_date = data.get("session_date")
    start_time = data.get("start_time")

    if not (course_code and course_name and instructor):
        return jsonify({
            "success": False,
            "error": "Course code, course name, and instructor are required.",
        }), 400

    try:
        session_id = db.create_session(
            course_code=course_code,
            course_name=course_name,
            instructor=instructor,
            session_date=session_date,
            start_time=start_time,
        )
        new_session = db.get_session(session_id)
        return jsonify({
            "success": True,
            "message": "Attendance session started.",
            "session": new_session,
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@sessions_bp.route("/api/sessions/active", methods=["GET"])
def get_current_active_session():
    """Retrieve the currently active attendance session."""
    active = db.get_active_session()
    if not active:
        return jsonify({"success": True, "active_session": None})

    # Include presence counts
    stats = db.get_dashboard_stats(session_id=active["session_id"])
    active_data = dict(active)
    active_data["stats"] = stats

    return jsonify({"success": True, "active_session": active_data})


@sessions_bp.route("/api/sessions/<int:session_id>/close", methods=["POST"])
def end_session(session_id: int):
    """Mark an active session as completed."""
    session = db.get_session(session_id)
    if not session:
        return jsonify({"success": False, "error": "Session not found."}), 404

    success = db.close_session(session_id)
    if not success:
        return jsonify({"success": False, "error": "Failed to close session."}), 500

    return jsonify({"success": True, "message": "Session completed successfully."})

