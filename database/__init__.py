"""FaceAttend AI - Database Package.

Handles SQLite connection pooling, schema initialization, CRUD operations,
and binary serialization for facial embeddings.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

