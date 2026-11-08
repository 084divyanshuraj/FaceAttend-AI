"""FaceAttend AI - Core Computer Vision Package.

Contains decoupled modules for face detection, alignment, recognition,
face quality assessment, vector matching, and temporal attendance tracking.
"""

from pathlib import Path
import sys

# Ensure project root is available for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

