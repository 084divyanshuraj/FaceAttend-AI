"""FaceAttend AI - Student Face Registration & Enrollment Engine.

Orchestrates multi-sample face acquisition, biometric quality gating,
diversity validation, database persistence, and in-memory vector cache synchronization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from core.detector import FaceDetection, YuNetDetector
from core.matcher import GLOBAL_MATCHER_CACHE
from core.quality import FaceQualityAssessor, QualityAssessmentResult
from core.recognizer import SFaceRecognizer
from database import db


@dataclass
class EnrollmentSessionState:
    """Tracks state and accumulated biometric samples during enrollment."""
    student_id: int
    target_samples: int = config.QUALITY_CONFIG["samples_per_student"]
    collected_embeddings: List[np.ndarray] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    is_complete: bool = False

    @property
    def samples_count(self) -> int:
        return len(self.collected_embeddings)


@dataclass
class EnrollmentFrameResult:
    """Feedback from evaluating a single video frame for enrollment."""
    status: str                          # "ACCEPTED", "REJECTED", "NO_FACE", "MULTIPLE_FACES", "COMPLETED"
    samples_collected: int
    samples_required: int
    message: str
    quality_result: Optional[QualityAssessmentResult] = None
    detection: Optional[FaceDetection] = None


class RegistrationEngine:
    """Coordinates detection, quality validation, alignment, and vector enrollment."""

    def __init__(
        self,
        detector: Optional[YuNetDetector] = None,
        recognizer: Optional[SFaceRecognizer] = None,
        assessor: Optional[FaceQualityAssessor] = None,
    ):
        self.detector = detector or YuNetDetector()
        self.recognizer = recognizer or SFaceRecognizer()
        self.assessor = assessor or FaceQualityAssessor()

    def process_frame(
        self,
        frame: np.ndarray,
        session: EnrollmentSessionState,
    ) -> EnrollmentFrameResult:
        """Evaluate a camera frame, enforce quality rules, and collect an embedding sample.

        Args:
            frame: Full BGR camera image.
            session: Active EnrollmentSessionState for the student.

        Returns:
            EnrollmentFrameResult with visual feedback and enrollment progress.
        """
        if session.is_complete:
            return EnrollmentFrameResult(
                status="COMPLETED",
                samples_collected=session.samples_count,
                samples_required=session.target_samples,
                message="Enrollment is already complete. All samples collected.",
            )

        # 1. Detect faces in frame
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return EnrollmentFrameResult(
                status="NO_FACE",
                samples_collected=session.samples_count,
                samples_required=session.target_samples,
                message="No face detected. Please position yourself in front of the camera.",
            )

        if len(detections) > 1:
            return EnrollmentFrameResult(
                status="MULTIPLE_FACES",
                samples_collected=session.samples_count,
                samples_required=session.target_samples,
                message="Multiple faces detected. Only the student being registered should be in frame.",
            )

        # Exactly 1 face detected
        detection = detections[0]

        # 2. Evaluate biometric quality
        quality_res = self.assessor.evaluate(frame, detection)
        if not quality_res.is_valid:
            return EnrollmentFrameResult(
                status="REJECTED",
                samples_collected=session.samples_count,
                samples_required=session.target_samples,
                message=quality_res.feedback_message,
                quality_result=quality_res,
                detection=detection,
            )

        # 3. Align face and extract 128-D embedding
        aligned_face = self.recognizer.align_crop(frame, detection)
        embedding = self.recognizer.extract_feature(aligned_face, normalize=True)

        # 4. Diversity Validation (prevent storing identical frames)
        if session.collected_embeddings:
            # Check maximum similarity with already captured samples
            max_sim = max(
                self.recognizer.compute_similarity(embedding, existing_emb)
                for existing_emb in session.collected_embeddings
            )
            # If similarity is extremely high (> 0.98), request slight head/expression shift
            if max_sim > 0.98:
                return EnrollmentFrameResult(
                    status="REJECTED",
                    samples_collected=session.samples_count,
                    samples_required=session.target_samples,
                    message="Sample too similar to previous. Please tilt or turn your head slightly.",
                    quality_result=quality_res,
                    detection=detection,
                )

        # 5. Accept and store sample in active session
        session.collected_embeddings.append(embedding)
        session.quality_scores.append(quality_res.quality_score)

        if session.samples_count >= session.target_samples:
            session.is_complete = True
            status = "COMPLETED"
            msg = f"Enrollment complete! All {session.target_samples} face samples successfully captured."
        else:
            status = "ACCEPTED"
            msg = f"Sample {session.samples_count} of {session.target_samples} captured successfully!"

        return EnrollmentFrameResult(
            status=status,
            samples_collected=session.samples_count,
            samples_required=session.target_samples,
            message=msg,
            quality_result=quality_res,
            detection=detection,
        )

    def commit_enrollment(
        self,
        session: EnrollmentSessionState,
        db_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Persist collected embeddings to SQLite and warm in-memory vector cache.

        Args:
            session: Fully completed EnrollmentSessionState.
            db_path: Optional custom database path.

        Returns:
            True if successfully saved and synced to memory cache.
        """
        if not session.collected_embeddings:
            return False

        # 1. Save to SQLite database
        db.save_face_embeddings(
            student_id=session.student_id,
            embeddings=session.collected_embeddings,
            quality_scores=session.quality_scores,
            db_path=db_path,
        )

        # 2. Retrieve student metadata
        student = db.get_student(session.student_id, db_path=db_path)
        if not student:
            return False

        full_name = f"{student['first_name']} {student['last_name']}".strip()

        # 3. Add to live in-memory vector cache
        GLOBAL_MATCHER_CACHE.add_student_samples(
            student_id=student["student_id"],
            roll_number=student["roll_number"],
            name=full_name,
            department=student["department"],
            embeddings=session.collected_embeddings,
        )

        return True

