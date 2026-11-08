"""FaceAttend AI - Unit & Verification Tests for Core Computer Vision Pipeline.

Tests model loading, detector execution, SFace alignment, feature extraction,
and cosine similarity matching.
"""

import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

import config
from core.detector import FaceDetection, YuNetDetector
from core.recognizer import SFaceRecognizer


def test_models_exist():
    """Verify that required ONNX models exist on disk."""
    assert config.YUNET_MODEL_PATH.exists(), f"YuNet model missing at {config.YUNET_MODEL_PATH}"
    assert config.SFACE_MODEL_PATH.exists(), f"SFace model missing at {config.SFACE_MODEL_PATH}"
    assert config.YUNET_MODEL_PATH.stat().st_size > 100 * 1024
    assert config.SFACE_MODEL_PATH.stat().st_size > 10 * 1024 * 1024


def test_detector_initialization():
    """Verify YuNet detector initializes correctly."""
    detector = YuNetDetector()
    assert detector is not None

    # Test on blank frame (should detect 0 faces)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list)
    assert len(detections) == 0

    # Test dynamic input size adaptation
    hd_blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections_hd = detector.detect(hd_blank)
    assert len(detections_hd) == 0


def test_recognizer_feature_extraction_and_matching():
    """Verify SFace feature extraction, normalization, and similarity matching."""
    recognizer = SFaceRecognizer()
    assert recognizer is not None

    # Test feature extraction on canonical 112x112 input
    dummy_face = np.full((112, 112, 3), 120, dtype=np.uint8)
    feat1 = recognizer.extract_feature(dummy_face, normalize=True)

    # 1. Output shape must be 128
    assert feat1.shape == (128,), f"Expected shape (128,), got {feat1.shape}"
    assert feat1.dtype == np.float32

    # 2. L2 norm must be approximately 1.0
    norm1 = np.linalg.norm(feat1)
    assert np.isclose(norm1, 1.0, atol=1e-4), f"Normalized vector norm={norm1} != 1.0"

    # 3. Self-similarity must be 1.0
    self_sim = recognizer.compute_similarity(feat1, feat1)
    assert np.isclose(self_sim, 1.0, atol=1e-4), f"Self similarity={self_sim} != 1.0"

    # 4. Self match check
    is_same, score = recognizer.is_match(feat1, feat1)
    assert is_same is True
    assert np.isclose(score, 1.0, atol=1e-4)

    # 5. Cosine distance between identical vectors must be 0.0
    dist = recognizer.compute_distance(feat1, feat1)
    assert np.isclose(dist, 0.0, atol=1e-4)

    # 6. Orthogonal vectors must yield 0.0 similarity
    v_ortho_1 = np.zeros(128, dtype=np.float32)
    v_ortho_1[0] = 1.0
    v_ortho_2 = np.zeros(128, dtype=np.float32)
    v_ortho_2[1] = 1.0
    assert np.isclose(recognizer.compute_similarity(v_ortho_1, v_ortho_2), 0.0, atol=1e-5)
    is_ortho_match, ortho_score = recognizer.is_match(v_ortho_1, v_ortho_2)
    assert is_ortho_match is False
    assert np.isclose(ortho_score, 0.0, atol=1e-5)

    # 7. Opposite vectors must yield -1.0 similarity
    assert np.isclose(recognizer.compute_similarity(v_ortho_1, -v_ortho_1), -1.0, atol=1e-5)


def test_cv_engine_latency():
    """Measure raw inference latency for face feature extraction."""
    recognizer = SFaceRecognizer()
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = recognizer.extract_feature(dummy_face)

    # Benchmark 30 runs
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = recognizer.extract_feature(dummy_face)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_latency_ms = np.mean(times)
    fps = 1000.0 / avg_latency_ms
    print(f"\n   [BENCHMARK] SFace Feature Extraction: {avg_latency_ms:.2f} ms ({fps:.1f} FPS on CPU)")
    assert avg_latency_ms < 150.0, f"Latency {avg_latency_ms:.2f}ms is higher than expected."


if __name__ == "__main__":
    print("=" * 60)
    print("Running Core Computer Vision Unit Tests")
    print("=" * 60)
    test_models_exist()
    print("[PASS] test_models_exist")
    test_detector_initialization()
    print("[PASS] test_detector_initialization")
    test_recognizer_feature_extraction_and_matching()
    print("[PASS] test_recognizer_feature_extraction_and_matching")
    test_cv_engine_latency()
    print("[PASS] test_cv_engine_latency")
    print("=" * 60)
    print("ALL CORE COMPUTER VISION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
