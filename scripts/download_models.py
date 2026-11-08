"""FaceAttend AI - Model Downloader Script.

Downloads the official pretrained ONNX models from the OpenCV Zoo repository:
1. YuNet (face_detection_yunet_2023mar.onnx) - Face Detection
2. SFace (face_recognition_sface_2021dec.onnx) - Face Recognition
"""

import sys
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

MODELS = {
    "YuNet Face Detector": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": config.YUNET_MODEL_PATH,
        "min_size_kb": 200,
    },
    "SFace Face Recognizer": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "path": config.SFACE_MODEL_PATH,
        "min_size_kb": 1000,
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100.0)
            sys.stdout.write(
                f"\r   [{percent:5.1f}%] {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB"
            )
        else:
            sys.stdout.write(f"\r   Downloaded {downloaded / 1024 / 1024:.2f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=progress_callback)
        print()
        temp_path.replace(dest_path)
        return True
    except Exception as e:
        print(f"\n   [ERROR] Failed to download from {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_models() -> bool:
    """Check if all models exist and are valid, downloading missing ones."""
    all_ok = True
    print("=" * 60)
    print("FaceAttend AI - Model Verification & Download")
    print("=" * 60)

    for name, info in MODELS.items():
        path: Path = info["path"]
        min_size = info["min_size_kb"] * 1024

        if path.exists() and path.stat().st_size >= min_size:
            print(f"[OK] {name} already present at {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"[DOWNLOADING] {name} from OpenCV Zoo...")
            success = download_file(info["url"], path)
            if success and path.exists() and path.stat().st_size >= min_size:
                print(f"[SUCCESS] {name} saved successfully ({path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"[FAILED] Could not acquire {name}.")
                all_ok = False

    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = ensure_models()
    sys.exit(0 if success else 1)

