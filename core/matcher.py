"""FaceAttend AI - High-Speed In-Memory Vector Matcher Cache.

Caches registered student embeddings as a contiguous unit-normalized NumPy matrix.
Performs sub-millisecond 1-to-N cosine similarity matching via BLAS matrix-vector
dot products.
"""

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

import numpy as np

import config
from database import db


@dataclass
class MatchResult:
    """Standardized face recognition match result."""
    is_known: bool
    student_id: Optional[int]
    roll_number: Optional[str]
    name: Optional[str]
    department: Optional[str]
    confidence: float
    label: str


class VectorMatcherCache:
    """Thread-safe in-memory vector cache and matcher."""

    def __init__(
        self,
        threshold: float = config.RECOGNITION_CONFIG["cosine_similarity_threshold"],
    ):
        self.threshold = float(threshold)
        self._lock = threading.Lock()
        self.embeddings_matrix: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.metadata)

    def warm_from_db(self, db_path: Optional[Union[str, Path]] = None) -> int:
        """Load and normalize all active student embeddings from SQLite into memory.

        Returns:
            Total number of loaded embedding samples.
        """
        records = db.get_all_embeddings_with_metadata(db_path)
        with self._lock:
            if not records:
                self.embeddings_matrix = np.empty((0, 128), dtype=np.float32)
                self.metadata = []
                return 0

            vectors = []
            meta_list = []
            for r in records:
                vec = r["embedding"].flatten().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                vectors.append(vec)
                meta_list.append({
                    "student_id": r["student_id"],
                    "roll_number": r["roll_number"],
                    "name": r["name"],
                    "department": r["department"],
                })

            self.embeddings_matrix = np.vstack(vectors).astype(np.float32)
            self.metadata = meta_list
            return len(self.metadata)

    def add_student_samples(
        self,
        student_id: int,
        roll_number: str,
        name: str,
        department: str,
        embeddings: List[np.ndarray],
    ) -> None:
        """Dynamically add newly enrolled student embeddings to the cache."""
        if not embeddings:
            return

        new_vectors = []
        new_meta = []
        for emb in embeddings:
            vec = emb.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            new_vectors.append(vec)
            new_meta.append({
                "student_id": student_id,
                "roll_number": roll_number,
                "name": name,
                "department": department,
            })

        new_matrix = np.vstack(new_vectors).astype(np.float32)

        with self._lock:
            if len(self.metadata) == 0:
                self.embeddings_matrix = new_matrix
                self.metadata = new_meta
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_matrix])
                self.metadata.extend(new_meta)

    def remove_student(self, student_id: int) -> None:
        """Dynamically remove a deleted student from the in-memory cache."""
        with self._lock:
            if len(self.metadata) == 0:
                return

            keep_indices = [
                i for i, m in enumerate(self.metadata) if m["student_id"] != student_id
            ]

            if len(keep_indices) == 0:
                self.embeddings_matrix = np.empty((0, 128), dtype=np.float32)
                self.metadata = []
            elif len(keep_indices) < len(self.metadata):
                self.embeddings_matrix = self.embeddings_matrix[keep_indices]
                self.metadata = [self.metadata[i] for i in keep_indices]

    def match(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Match a single face query embedding against the cached student gallery.

        Args:
            query_embedding: 128-dimensional embedding from SFace.
            threshold: Optional custom threshold.

        Returns:
            MatchResult with identification details or Unknown.
        """
        thresh = self.threshold if threshold is None else float(threshold)

        # Normalize query vector to unit length
        q = query_embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm

        with self._lock:
            if self.embeddings_matrix.shape[0] == 0:
                return MatchResult(
                    is_known=False,
                    student_id=None,
                    roll_number=None,
                    name=None,
                    department=None,
                    confidence=0.0,
                    label="Unknown",
                )

            # Compute vectorized dot product: S = E . q (Shape: (M,))
            scores = np.dot(self.embeddings_matrix, q)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= thresh:
                meta = self.metadata[best_idx]
                return MatchResult(
                    is_known=True,
                    student_id=meta["student_id"],
                    roll_number=meta["roll_number"],
                    name=meta["name"],
                    department=meta["department"],
                    confidence=best_score,
                    label=f"{meta['name']} ({meta['roll_number']})",
                )
            else:
                return MatchResult(
                    is_known=False,
                    student_id=None,
                    roll_number=None,
                    name=None,
                    department=None,
                    confidence=best_score,
                    label="Unknown",
                )


# Global singleton instance for application-wide vector caching
GLOBAL_MATCHER_CACHE = VectorMatcherCache()

