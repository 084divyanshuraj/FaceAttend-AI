"""FaceAttend AI - Live Attendance & Frame Processing Blueprint.

Processes incoming client-side camera frames via WebSocket/REST, performs
real-time multi-face identification, and manages manual attendance overrides.
"""

import base64
from typing import Optional
import cv2
from flask import Blueprint, jsonify, request
import numpy as np

from core.attendance_engine import AttendanceEngine
from database import db

attendance_bp = Blueprint("attendance", __name__)
ATTENDANCE_ENGINE = AttendanceEngine()


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 data URI to OpenCV BGR image."""
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


@attendance_bp.route("/api/attendance/process-frame", methods=["POST"])
def process_live_frame():
    """Analyze a single camera frame for real-time multi-face recognition and attendance logging."""
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    session_id = data.get("session_id")

    if not image_base64:
        return jsonify({"success": False, "error": "No image data provided."}), 400

    frame = decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"success": False, "error": "Invalid image payload."}), 400

    # Process frame through Attendance Engine
    res = ATTENDANCE_ENGINE.process_frame(
        frame=frame,
        session_id=session_id,
        draw_annotations=False,  # Frontend handles rendering via HTML5 canvas
    )

    return jsonify({
        "success": True,
        "timestamp": res.timestamp,
        "faces_detected": res.faces_detected,
        "faces": [f.to_dict() for f in res.faces],
        "new_attendance_events": res.new_attendance_events,
        "session_stats": res.session_stats,
    })


@attendance_bp.route("/api/attendance/session/<int:session_id>", methods=["GET"])
def get_session_attendance_list(session_id: int):
    """Retrieve full attendance list for a specific session."""
    session_data = db.get_session(session_id)
    if not session_data:
        return jsonify({"success": False, "error": "Session not found."}), 404

    records = db.get_session_attendance(session_id)
    return jsonify({
        "success": True,
        "session": session_data,
        "attendance": records,
        "count": len(records),
    })


@attendance_bp.route("/api/attendance/override", methods=["POST"])
def manual_attendance_override():
    """Perform an administrative manual attendance override."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    student_id = data.get("student_id")
    action = data.get("action", "").upper()  # "MARK_PRESENT" or "REMOVE"
    remarks = data.get("remarks", "Manual admin override")
    admin_user = data.get("admin_user", "admin")

    if not (session_id and student_id and action):
        return jsonify({
            "success": False,
            "error": "session_id, student_id, and action are required.",
        }), 400

    try:
        success = db.manual_attendance_override(
            session_id=int(session_id),
            student_id=int(student_id),
            action=action,
            admin_user=admin_user,
            remarks=remarks,
        )
        # Sync attendance engine session state
        ATTENDANCE_ENGINE.sync_session_state(int(session_id))

        return jsonify({
            "success": True,
            "message": f"Successfully performed override: {action}.",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

