"""FaceAttend AI - SFace Face Recognizer Module.

Wraps OpenCV's native FaceRecognizerSF for 5-point landmark-based face alignment
and 128-dimensional deep metric embedding extraction.
Supports both CPU (local/deploy) and CUDA (Kaggle GPU) backends.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

import config
from core.detector import FaceDetection


class SFaceRecognizer:
    """SFace / Custom ArcFace Feature Extractor and Aligner.
    Supports both pretrained SFace and custom Kaggle fine-tuned ONNX models.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        cosine_threshold: float = config.RECOGNITION_CONFIG["cosine_similarity_threshold"],
        backend: str = config.COMPUTE_BACKEND,
    ):
        # Determine model path (Custom Fine-tuned or SFace)
        if model_path:
            chosen_path = Path(model_path)
        elif config.ACTIVE_RECOGNITION_MODEL == "CUSTOM" and config.CUSTOM_MODEL_PATH.exists():
            chosen_path = config.CUSTOM_MODEL_PATH
        else:
            chosen_path = config.SFACE_MODEL_PATH

        self.model_path = chosen_path
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please run 'python scripts/download_models.py' first."
            )

        self.cosine_threshold = float(cosine_threshold)
        self.backend_str = backend.upper()

        if self.backend_str == "CUDA":
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA
        else:
            backend_id = cv2.dnn.DNN_BACKEND_OPENCV
            target_id = cv2.dnn.DNN_TARGET_CPU

        # SFace model is always used for canonical 112x112 landmark alignment
        self._aligner = cv2.FaceRecognizerSF.create(
            model=str(config.SFACE_MODEL_PATH),
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )

        # For feature extraction: if custom ONNX, use OpenCV DNN Net; else FaceRecognizerSF
        self.is_custom = (self.model_path.name != config.SFACE_MODEL_PATH.name)
        if self.is_custom:
            self._custom_net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._custom_net.setPreferableBackend(backend_id)
            self._custom_net.setPreferableTarget(target_id)
            self.recognizer = None
        else:
            self._custom_net = None
            self.recognizer = self._aligner

    def align_crop(
        self,
        image: np.ndarray,
        detection: Union[FaceDetection, np.ndarray],
    ) -> np.ndarray:
        """Align and crop a detected face to 112x112 canonical space.

        Args:
            image: Original full-frame BGR image.
            detection: Either a FaceDetection instance or raw 15-element numpy array.

        Returns:
            Aligned face crop of shape (112, 112, 3), dtype uint8.
        """
        raw_face = detection.raw if isinstance(detection, FaceDetection) else detection
        if raw_face is None or len(raw_face) < 15:
            raise ValueError("Invalid face detection data supplied for alignment.")

        aligned_face = self._aligner.alignCrop(image, raw_face)
        return aligned_face

    def extract_feature(
        self,
        aligned_face: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract a 128-dimensional facial embedding vector.

        Supports both standard SFace and custom Kaggle fine-tuned ONNX models.

        Args:
            aligned_face: Aligned face crop (112, 112, 3).
            normalize: If True, normalizes the embedding to unit L2 norm.

        Returns:
            1D float32 numpy array of shape (128,).
        """
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Cannot extract features from an empty face crop.")

        if self.is_custom and self._custom_net is not None:
            # Preprocess for custom PyTorch MobileFaceNet ONNX model (1x3x112x112)
            blob = cv2.dnn.blobFromImage(
                aligned_face,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            )
            self._custom_net.setInput(blob)
            feature = self._custom_net.forward()
        else:
            feature = self.recognizer.feature(aligned_face)

        feature = feature.flatten().astype(np.float32)

        if normalize:
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm

        return feature

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute Cosine Similarity between two normalized embeddings.

        Returns:
            Cosine similarity score in range [-1.0, 1.0]. 1.0 indicates identical.
        """
        v1 = feat1.flatten().astype(np.float32)
        v2 = feat2.flatten().astype(np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(-1.0, min(1.0, similarity))

    def compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute Cosine Distance (1 - Cosine Similarity).

        Returns:
            Cosine distance in range [0.0, 2.0]. 0.0 indicates identical.
        """
        return 1.0 - self.compute_similarity(feat1, feat2)

    def is_match(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Determine if two embeddings belong to the same person.

        Args:
            feat1: First face embedding.
            feat2: Second face embedding.
            threshold: Optional custom threshold; defaults to calibrated threshold (0.637).

        Returns:
            Tuple of (is_same_identity: bool, cosine_similarity: float).
        """
        thresh = self.cosine_threshold if threshold is None else threshold
        similarity = self.compute_similarity(feat1, feat2)
        return (similarity >= thresh), similarity

