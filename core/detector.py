"""FaceAttend AI - YuNet Face Detector Module.

Wraps OpenCV's native FaceDetectorYN for lightweight, real-time face detection
and 5-point facial landmark estimation.
Supports both CPU (local/deploy) and CUDA (Kaggle GPU) backends.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

import config


@dataclass
class FaceDetection:
    """Standardized face detection result."""
    bbox: Tuple[int, int, int, int]      # (x, y, width, height) in integer pixels
    confidence: float                    # Detection confidence score [0.0 - 1.0]
    landmarks: np.ndarray                # 5 landmarks of shape (5, 2): [right_eye, left_eye, nose, right_mouth, left_mouth]
    raw: np.ndarray                      # Raw 15-element array directly compatible with SFace alignCrop


class YuNetDetector:
    """YuNet Face Detector wrapper around cv2.FaceDetectorYN."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        score_threshold: float = config.DETECTION_CONFIG["score_threshold"],
        nms_threshold: float = config.DETECTION_CONFIG["nms_threshold"],
        top_k: int = config.DETECTION_CONFIG["top_k"],
        backend: str = config.COMPUTE_BACKEND,
    ):
        self.model_path = Path(model_path or config.YUNET_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at '{self.model_path}'. "
                f"Please run 'python scripts/download_models.py' first."
            )

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.backend_str = backend.upper()

        # Map backend string to OpenCV DNN constants
        if self.backend_str == "CUDA":
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA
        else:
            backend_id = cv2.dnn.DNN_BACKEND_OPENCV
            target_id = cv2.dnn.DNN_TARGET_CPU

        # Initial input size placeholder (width, height)
        self._current_input_size = config.DETECTION_CONFIG["default_input_size"]

        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=self._current_input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=backend_id,
            target_id=target_id,
        )

    def _ensure_input_size(self, width: int, height: int):
        """Update detector input size dynamically if frame dimensions change."""
        if (width, height) != self._current_input_size:
            self.detector.setInputSize((width, height))
            self._current_input_size = (width, height)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a BGR image frame.

        Args:
            image: numpy ndarray of shape (H, W, 3) in BGR color space.

        Returns:
            List of FaceDetection objects, sorted by confidence descending.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self._ensure_input_size(w, h)

        # OpenCV returns tuple: (retval, faces)
        retval, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        detections: List[FaceDetection] = []
        for face in faces:
            # Face array format (15 elements):
            # [0:4] = bbox (x, y, w, h)
            # [4:14] = 5 landmarks (re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y)
            # [14] = confidence score
            x, y, bw, bh = face[0:4]
            conf = float(face[14])
            landmarks = face[4:14].reshape((5, 2))

            # Clamp bounding box coordinates to frame boundaries
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            bw = max(0, min(int(round(bw)), w - x1))
            bh = max(0, min(int(round(bh)), h - y1))

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, bw, bh),
                    confidence=conf,
                    landmarks=landmarks,
                    raw=face,
                )
            )

        return detections

