"""FaceAttend AI - Analytics & CSV Reporting Blueprint.

Provides aggregate dashboard statistics, Chart.js metrics, and CSV report export.
"""

import csv
import io
from flask import Blueprint, Response, jsonify, request
from database import db

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/api/analytics/dashboard", methods=["GET"])
def get_dashboard_summary():
    """Retrieve high-level statistics for administrative dashboards."""
    active_session = db.get_active_session()
    session_id = active_session["session_id"] if active_session else None

    stats = db.get_dashboard_stats(session_id=session_id)

    with db.get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS total_sessions FROM sessions")
        stats["total_sessions"] = cursor.fetchone()["total_sessions"]

        cursor.execute("SELECT COUNT(*) AS total_records FROM attendance")
        stats["total_attendance_logs"] = cursor.fetchone()["total_records"]

    stats["active_session"] = active_session
    return jsonify({"success": True, "stats": stats})


@analytics_bp.route("/api/analytics/daily-trend", methods=["GET"])
def get_daily_attendance_trend():
    """Retrieve daily attendance counts for the last 7 sessions for charting."""
    sql = """
        SELECT
            s.session_date,
            s.course_code,
            COUNT(a.attendance_id) AS present_count
        FROM sessions s
        LEFT JOIN attendance a ON s.session_id = a.session_id
        GROUP BY s.session_id
        ORDER BY s.session_date DESC, s.session_id DESC
        LIMIT 10
    """
    with db.get_db_cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        # Reverse to show chronological order
        data = [dict(r) for r in reversed(rows)]
        return jsonify({"success": True, "trend": data})


@analytics_bp.route("/api/analytics/department-breakdown", methods=["GET"])
def get_department_breakdown():
    """Retrieve attendance count categorized by academic department."""
    sql = """
        SELECT
            s.department,
            COUNT(a.attendance_id) AS attendance_count
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id
        GROUP BY s.department
        ORDER BY attendance_count DESC
    """
    with db.get_db_cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return jsonify({"success": True, "departments": [dict(r) for r in rows]})


@analytics_bp.route("/api/reports/export-csv", methods=["GET"])
def export_attendance_csv():
    """Generate and stream a CSV report of attendance records."""
    session_id = request.args.get("session_id")
    department = request.args.get("department")

    where_clauses = []
    params = []

    if session_id:
        where_clauses.append("a.session_id = ?")
        params.append(session_id)

    if department:
        where_clauses.append("s.department = ?")
        params.append(department.strip().upper())

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
        SELECT
            s.roll_number,
            s.first_name || ' ' || s.last_name AS student_name,
            s.department,
            sess.course_code,
            sess.course_name,
            sess.session_date,
            a.marked_at,
            a.method,
            ROUND(a.confidence_score * 100, 1) || '%' AS confidence,
            COALESCE(a.remarks, '') AS remarks
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        JOIN sessions sess ON a.session_id = sess.session_id
        {where_sql}
        ORDER BY a.marked_at DESC
    """

    with db.get_db_cursor() as cursor:
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

    # Generate CSV stream in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Roll Number",
        "Student Name",
        "Department",
        "Course Code",
        "Course Name",
        "Session Date",
        "Timestamp",
        "Method",
        "Confidence",
        "Remarks",
    ])

    for r in rows:
        writer.writerow(list(r))

    output.seek(0)
    filename = f"faceattend_report_{session_id or 'all'}.csv"

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

