"""FaceAttend AI - Verification Test for Custom Fine-Tuned ArcFace ONNX Model.

Verifies that when a custom trained ONNX model is provided (e.g. from Kaggle),
SFaceRecognizer dynamically loads the custom graph and extracts 128-D normalized embeddings.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

import config
from core.recognizer import SFaceRecognizer
from scripts.train_model import MobileFaceNet


def test_custom_onnx_model_export_and_inference():
    """Verify export of custom MobileFaceNet to ONNX and inference via SFaceRecognizer."""
    test_onnx_path = config.MODELS_DIR / "test_custom_model.onnx"

    # 1. Export temporary PyTorch model to ONNX
    model = MobileFaceNet(embedding_size=128)
    model.eval()
    dummy = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy,
        str(test_onnx_path),
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["embedding"],
        dynamo=False,
    )
    assert test_onnx_path.exists()

    # 2. Initialize SFaceRecognizer pointing to custom ONNX model
    recognizer = SFaceRecognizer(model_path=test_onnx_path)
    assert recognizer.is_custom is True

    # 3. Test feature extraction on canonical 112x112 input
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)
    feat = recognizer.extract_feature(dummy_face, normalize=True)

    assert feat.shape == (128,)
    assert feat.dtype == np.float32
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0, atol=1e-4)

    # Clean up test file
    if test_onnx_path.exists():
        test_onnx_path.unlink()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Custom Trained Model Integration Test")
    print("=" * 60)
    test_custom_onnx_model_export_and_inference()
    print("[PASS] test_custom_onnx_model_export_and_inference")
    print("=" * 60)
    print("CUSTOM ONNX INTEGRATION VERIFIED SUCCESSFULLY!")
    print("=" * 60)
