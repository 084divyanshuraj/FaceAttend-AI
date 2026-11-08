"""FaceAttend AI - Central Configuration Module.

Defines all filesystem paths, Computer Vision model thresholds,
database configurations, and session parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "faceattend.db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure required directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Model Filepaths
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
CUSTOM_MODEL_PATH = MODELS_DIR / "custom_face_recognizer.onnx"

# Active Model Selection: "SFACE" (default) or "CUSTOM" (Kaggle fine-tuned)
ACTIVE_RECOGNITION_MODEL = "CUSTOM" if CUSTOM_MODEL_PATH.exists() else "SFACE"

# Compute Backend Configuration
# Supports CPU for local development/deployment, and CUDA for Kaggle GPU environments.
COMPUTE_BACKEND = "CPU"  # "CPU" or "CUDA"

# YuNet Detection Parameters
DETECTION_CONFIG = {
    "score_threshold": 0.8,    # Confidence score threshold for face detection
    "nms_threshold": 0.3,      # Non-Maximum Suppression threshold
    "top_k": 50,               # Maximum number of faces to detect per frame
    "default_input_size": (640, 640),  # (width, height)
    "backend": COMPUTE_BACKEND,
}

# SFace Recognition Parameters
# Thresholds calibrated according to OpenCV Zoo benchmarks:
# - Cosine Similarity: >= 0.637 denotes the same identity
# - Cosine Distance: <= 0.363 denotes the same identity
# - L2 Norm Distance: <= 1.128 denotes the same identity
RECOGNITION_CONFIG = {
    "cosine_similarity_threshold": 0.637,
    "cosine_distance_threshold": 0.363,
    "l2_distance_threshold": 1.128,
    "embedding_dimension": 128,
}

# Face Quality Assessment (FQA) Parameters for Enrollment
QUALITY_CONFIG = {
    "min_face_width": 80,         # Minimum face width in pixels
    "min_face_height": 80,        # Minimum face height in pixels
    "blur_threshold": 80.0,       # Minimum variance of Laplacian for sharpness
    "max_pose_angle_deg": 20.0,   # Maximum allowed eye-line tilt angle
    "samples_per_student": 5,     # Required number of face samples during enrollment
}

# Attendance Session Parameters
ATTENDANCE_CONFIG = {
    "temporal_window_size": 5,    # Number of recent frames tracked for consensus
    "temporal_consensus_count": 3,# Required detections of same ID within the window
    "session_cooldown_seconds": 60,# Minimum time between repeated attendance marks for same student
}

# Flask Application Settings
FLASK_CONFIG = {
    "SECRET_KEY": "faceattend-ai-dev-secret-key-change-in-prod",
    "DEBUG": True,
    "HOST": "0.0.0.0",
    "PORT": 5000,
}
