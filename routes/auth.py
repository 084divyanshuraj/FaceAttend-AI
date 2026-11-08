"""FaceAttend AI - Administrative Authentication Blueprint.

Provides session-based administrative login, logout, and access verification.
"""

from functools import wraps
from flask import Blueprint, jsonify, request, session

auth_bp = Blueprint("auth", __name__)

ADMIN_CREDENTIALS = {
    "username": "admin",
    "password": "admin123",  # Academic/dev default
}


def login_required(f):
    """Decorator to protect administrative endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"success": False, "error": "Unauthorized. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate administrator."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
        session["logged_in"] = True
        session["user"] = username
        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": username,
        })

    return jsonify({
        "success": False,
        "error": "Invalid username or password.",
    }), 401


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    """Clear administrative session."""
    session.clear()
    return jsonify({
        "success": True,
        "message": "Logged out successfully.",
    })


@auth_bp.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Check current authentication status."""
    is_authenticated = bool(session.get("logged_in"))
    return jsonify({
        "authenticated": is_authenticated,
        "user": session.get("user") if is_authenticated else None,
    })

