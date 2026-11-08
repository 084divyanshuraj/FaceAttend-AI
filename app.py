"""FaceAttend AI - Main Flask Application Factory & Server Entry Point.

Initializes database, warms in-memory vector cache, registers blueprints,
and provides view routes for modern web dashboards.
"""

from pathlib import Path
import sys

# Ensure project root is available for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, redirect, render_template, session, url_for
from flask_cors import CORS

import config
from core.matcher import GLOBAL_MATCHER_CACHE
from database import db
from routes.analytics import analytics_bp
from routes.attendance import attendance_bp
from routes.auth import auth_bp
from routes.sessions import sessions_bp
from routes.students import students_bp


def create_app() -> Flask:
    """Initialize and configure the Flask web application."""
    app = Flask(
        __name__,
        template_folder=str(config.TEMPLATES_DIR),
        static_folder=str(config.STATIC_DIR),
    )

    app.config["SECRET_KEY"] = config.FLASK_CONFIG["SECRET_KEY"]
    CORS(app)

    # 1. Initialize Database & Warm In-Memory Vector Matcher Cache
    db.init_db()
    enrolled_count = GLOBAL_MATCHER_CACHE.warm_from_db()
    print(f"[*] FaceAttend AI: Warmed vector cache with {enrolled_count} biometric samples.")

    # 2. Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(analytics_bp)

    # 3. View Routes (Frontend Dashboard Pages)
    @app.route("/")
    def index():
        return redirect(url_for("dashboard_view"))

    @app.route("/login")
    def login_view():
        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard_view():
        return render_template("dashboard.html")

    @app.route("/students")
    def students_view():
        return render_template("students.html")

    @app.route("/register")
    def register_view():
        return render_template("register.html")

    @app.route("/live")
    def live_attendance_view():
        return render_template("live.html")

    @app.route("/records")
    def records_view():
        return render_template("records.html")

    @app.route("/settings")
    def settings_view():
        return render_template("settings.html", config=config)

    # 4. Health & Status Route
    @app.route("/health")
    def health_check():
        active_sess = db.get_active_session()
        return jsonify({
            "status": "healthy",
            "enrolled_biometrics": len(GLOBAL_MATCHER_CACHE),
            "active_session": active_sess["session_code"] if active_sess else None,
            "backend": config.COMPUTE_BACKEND,
        })

    return app


if __name__ == "__main__":
    application = create_app()
    print(f"[*] Starting FaceAttend AI on http://127.0.0.1:{config.FLASK_CONFIG['PORT']}")
    application.run(
        host=config.FLASK_CONFIG["HOST"],
        port=config.FLASK_CONFIG["PORT"],
        debug=config.FLASK_CONFIG["DEBUG"],
    )

