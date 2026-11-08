"""FaceAttend AI - Standalone Desktop Kiosk & OpenCV Camera Runner.

Runs real-time multi-face attendance recognition directly via OpenCV VideoCapture,
displaying live bounding boxes, status badges, and an on-screen HUD without requiring
a web browser.
"""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

import config
from core.attendance_engine import AttendanceEngine
from core.detector import YuNetDetector
from core.matcher import GLOBAL_MATCHER_CACHE
from core.recognizer import SFaceRecognizer
from database import db


def main():
    parser = argparse.ArgumentParser(description="FaceAttend AI Desktop Kiosk Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--session", type=int, default=None, help="Session ID to record attendance for")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    args = parser.parse_args()

    print("=" * 60)
    print("FaceAttend AI - Desktop Kiosk Mode")
    print("=" * 60)

    # 1. Initialize Database & Warm Vector Cache
    db.init_db()
    count = GLOBAL_MATCHER_CACHE.warm_from_db()
    print(f"[*] Warmed in-memory vector gallery: {count} samples across {len(GLOBAL_MATCHER_CACHE.metadata)} students.")

    # 2. Determine Session
    active_session = db.get_active_session()
    session_id = args.session or (active_session["session_id"] if active_session else None)

    if session_id:
        sess_info = db.get_session(session_id)
        print(f"[*] Recording attendance for session {session_id}: {sess_info['course_code']} - {sess_info['course_name']}")
    else:
        print("[!] No active session specified. Running in visual recognition mode only.")

    # 3. Initialize CV Engine
    detector = YuNetDetector()
    recognizer = SFaceRecognizer()
    engine = AttendanceEngine(
        detector=detector,
        recognizer=recognizer,
        matcher_cache=GLOBAL_MATCHER_CACHE,
    )

    # 4. Open Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device {args.camera}.")
        return

    print("[*] Camera started. Press 'q' to quit.")
    fps_history = []

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received.")
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame through Attendance Engine
            result = engine.process_frame(
                frame=frame,
                session_id=session_id,
                draw_annotations=True,
            )

            annotated = result.annotated_frame or frame

            # Calculate FPS
            t1 = time.perf_counter()
            fps = 1.0 / max(1e-5, (t1 - t0))
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Draw HUD Header
            cv2.rectangle(annotated, (0, 0), (args.width, 36), (15, 23, 42), -1)
            cv2.line(annotated, (0, 36), (args.width, 36), (51, 65, 85), 1)

            hud_text = f"FaceAttend AI | FPS: {avg_fps:.1f} | Faces: {result.faces_detected}"
            if session_id:
                stats = result.session_stats
                hud_text += f" | Present: {stats.get('present_students', 0)}/{stats.get('total_students', 0)} ({stats.get('attendance_percentage', 0)}%)"

            cv2.putText(
                annotated,
                hud_text,
                (12, 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (248, 250, 252),
                1,
                cv2.LINE_AA,
            )

            # Print new events to terminal
            for evt in result.new_attendance_events:
                print(f"[ATTENDANCE RECORDED] {evt['name']} ({evt['roll_number']}) - Confidence: {evt['confidence']:.1%}")

            cv2.imshow("FaceAttend AI - Desktop Kiosk", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Kiosk session terminated.")


if __name__ == "__main__":
    main()

