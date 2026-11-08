"""FaceAttend AI - Face Quality Assessment (FQA) Module.

Evaluates detected faces for sharpness (Laplacian variance), head pose (roll and yaw),
resolution (minimum bounding box), illumination (brightness), and detector confidence.
Ensures only high-quality biometric samples are enrolled for student recognition.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from core.detector import FaceDetection


@dataclass
class QualityAssessmentResult:
    """Detailed quality assessment evaluation for a detected face."""
    is_valid: bool
    quality_score: float             # Normalized composite score [0.0 - 1.0]
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback_message: str = "Face quality is acceptable."


class FaceQualityAssessor:
    """Evaluates facial image quality against strict biometric standards."""

    def __init__(
        self,
        min_width: int = config.QUALITY_CONFIG["min_face_width"],
        min_height: int = config.QUALITY_CONFIG["min_face_height"],
        blur_threshold: float = config.QUALITY_CONFIG["blur_threshold"],
        max_pose_angle_deg: float = config.QUALITY_CONFIG["max_pose_angle_deg"],
        min_brightness: float = 40.0,
        max_brightness: float = 230.0,
        min_confidence: float = 0.85,
    ):
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.blur_threshold = float(blur_threshold)
        self.max_pose_angle_deg = float(max_pose_angle_deg)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.min_confidence = float(min_confidence)

    def compute_blur_score(self, face_crop: np.ndarray) -> float:
        """Calculate focus sharpness via variance of the Laplacian operator."""
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_brightness(self, face_crop: np.ndarray) -> float:
        """Calculate average luminance across the face region."""
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        return float(np.mean(gray))

    def compute_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Estimate head roll angle and horizontal yaw symmetry ratio.

        Landmarks order from YuNet:
        [0]: Right eye, [1]: Left eye, [2]: Nose tip, [3]: Right mouth, [4]: Left mouth

        Returns:
            (roll_angle_deg, yaw_ratio)
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return 0.0, 1.0

        re_x, re_y = landmarks[0]
        le_x, le_y = landmarks[1]
        nt_x, nt_y = landmarks[2]

        # 1. Roll (tilt angle): angle of eye line relative to horizontal
        dy = le_y - re_y
        dx = le_x - re_x
        roll_angle_deg = abs(math.degrees(math.atan2(dy, dx)))

        # 2. Yaw symmetry ratio: distance from nose tip to right eye vs left eye
        d_re = math.hypot(nt_x - re_x, nt_y - re_y)
        d_le = math.hypot(nt_x - le_x, nt_y - le_y)
        yaw_ratio = d_re / (d_le + 1e-6)

        return roll_angle_deg, yaw_ratio

    def evaluate(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> QualityAssessmentResult:
        """Perform comprehensive biometric quality assessment on a detected face.

        Args:
            frame: Full BGR frame.
            detection: FaceDetection object containing bbox, landmarks, and confidence.

        Returns:
            QualityAssessmentResult with pass/fail decision, issues list, and metrics.
        """
        x, y, w, h = detection.bbox
        frame_h, frame_w = frame.shape[:2]

        # Extract face crop
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return QualityAssessmentResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Invalid face boundary coordinates."],
                feedback_message="Face is outside the visible frame boundary.",
            )

        face_crop = frame[y1:y2, x1:x2]

        issues: List[str] = []
        metrics: Dict[str, float] = {}

        # 1. Resolution Check
        metrics["width"] = float(w)
        metrics["height"] = float(h)
        if w < self.min_width or h < self.min_height:
            issues.append(f"Face resolution too small ({w}x{h} px; minimum is {self.min_width}x{self.min_height} px).")

        # 2. Detection Confidence Check
        metrics["confidence"] = float(detection.confidence)
        if detection.confidence < self.min_confidence:
            issues.append(f"Detection confidence too low ({detection.confidence:.2f}; minimum is {self.min_confidence:.2f}).")

        # 3. Blur / Sharpness Check
        blur_score = self.compute_blur_score(face_crop)
        metrics["blur_score"] = blur_score
        if blur_score < self.blur_threshold:
            issues.append(f"Image is blurry (Laplacian variance {blur_score:.1f}; minimum is {self.blur_threshold:.1f}).")

        # 4. Illumination / Brightness Check
        brightness = self.compute_brightness(face_crop)
        metrics["brightness"] = brightness
        if brightness < self.min_brightness:
            issues.append(f"Face is too dark (brightness {brightness:.1f}; minimum is {self.min_brightness:.1f}).")
        elif brightness > self.max_brightness:
            issues.append(f"Face is overexposed/washed out (brightness {brightness:.1f}; maximum is {self.max_brightness:.1f}).")

        # 5. Head Pose Check (Roll and Yaw)
        roll_angle, yaw_ratio = self.compute_head_pose(detection.landmarks)
        metrics["roll_angle_deg"] = roll_angle
        metrics["yaw_ratio"] = yaw_ratio

        if roll_angle > self.max_pose_angle_deg:
            issues.append(f"Head is tilted ({roll_angle:.1f}°; maximum allowed is {self.max_pose_angle_deg:.1f}°).")

        # Extreme profile view check (yaw ratio outside 0.45 - 2.2)
        if yaw_ratio < 0.45 or yaw_ratio > 2.2:
            issues.append("Face is turned to the side; please look directly at the camera.")

        # Compute normalized composite quality score [0.0 - 1.0]
        # Sharpness score normalized up to 300
        norm_blur = min(1.0, blur_score / 300.0)
        norm_res = min(1.0, min(w / 160.0, h / 160.0))
        norm_pose = max(0.0, 1.0 - (roll_angle / self.max_pose_angle_deg))
        norm_conf = min(1.0, detection.confidence)

        composite_score = round(
            float(0.35 * norm_blur + 0.25 * norm_res + 0.20 * norm_pose + 0.20 * norm_conf),
            3,
        )

        is_valid = len(issues) == 0

        # Construct clear user feedback message
        if is_valid:
            feedback = "Face quality is excellent. Hold steady."
        else:
            # Highlight the most critical issue
            if "blurry" in issues[0].lower():
                feedback = "Image is blurry. Please hold steady."
            elif "tilted" in issues[0].lower() or "turned" in issues[0].lower():
                feedback = "Please face the camera directly."
            elif "too dark" in issues[0].lower():
                feedback = "Lighting is too dark. Increase ambient light."
            elif "resolution" in issues[0].lower():
                feedback = "Please move closer to the camera."
            else:
                feedback = issues[0]

        return QualityAssessmentResult(
            is_valid=is_valid,
            quality_score=composite_score,
            issues=issues,
            metrics=metrics,
            feedback_message=feedback,
        )

