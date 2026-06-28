"""Face detection and recognition service using DeepFace."""

import os
import pickle
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from deepface import DeepFace
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.services.media_scanner import fast_scan_for_media


class FaceService:
    """Service for face detection and similarity search.

    Maintains a database of known face embeddings and searches newly
    scanned images for matches.  The image-path cache is invalidated
    every ``CACHE_TTL`` seconds to pick up new files.

    Attributes:
        face_db_path: Path to the pickle file storing known-face embeddings.
        known_faces: Dict mapping person name → face embedding vector.
    """

    CACHE_TTL = 120  # seconds between re-scans

    def __init__(self, face_db_path: str = "face_db/known_faces.pkl", face_data_dir: str = "face_data"):
        self.face_db_path = face_db_path
        self.known_faces: Dict[str, list] = {}
        self._cached_paths: List[str] = []
        self._cache_timestamp: float = 0.0
        self.load_db()
        self.register_faces_from_directory(face_data_dir)

    # Database management

    def load_db(self):
        """Load known-face embeddings from disk.

        If the pickle file does not exist, an empty database is used.
        """
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, "rb") as f:
                    self.known_faces = pickle.load(f)
                logger.info("Loaded %d known faces", len(self.known_faces))
            except Exception as exc:
                logger.error("Error loading face database: %s", exc)
        else:
            logger.warning("Face database not found: %s", self.face_db_path)

    def save_db(self):
        """Persist known-face embeddings to disk."""
        os.makedirs(os.path.dirname(self.face_db_path) or ".", exist_ok=True)
        with open(self.face_db_path, "wb") as f:
            pickle.dump(self.known_faces, f)
        logger.info("Saved %d known faces", len(self.known_faces))

    # Registration

    def register_face(self, name: str, image_path: str) -> bool:
        """Encode a face from an image and add it to the known-faces database.

        Args:
            name: Person name (case-insensitive, stored lowercased).
            image_path: Path to a reference photo containing the face.

        Returns:
            True if a face was successfully encoded and stored.
        """
        try:
            embeddings = DeepFace.represent(
                img_path=image_path, model_name="Facenet", enforce_detection=False
            )
            if not embeddings:
                logger.warning("No face detected in %s", image_path)
                return False
            self.known_faces[name.lower().strip()] = embeddings[0]["embedding"]
            logger.info("Registered face for '%s' from %s", name, image_path)
            return True
        except Exception as exc:
            logger.error("Failed to register face for '%s': %s", name, exc)
            return False

    def register_faces_from_directory(self, directory: str = "face_data") -> int:
        """Register faces from images in a directory where the filename (minus
        extension) is used as the person name.

        Skips names that are already in the database.

        Args:
            directory: Path to the folder containing reference face images.

        Returns:
            Number of faces newly registered.
        """
        if not os.path.isdir(directory):
            return 0
        count = 0
        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            name = os.path.splitext(fname)[0].lower().strip()
            if name in self.known_faces:
                continue
            path = os.path.join(directory, fname)
            if self.register_face(name, path):
                count += 1
        if count > 0:
            self.save_db()
        return count

    # Search

    def search_by_name(self, name_query: str, threshold: float = 0.75) -> List[str]:
        """Find images containing a known person.

        Args:
            name_query: Person name (case-insensitive lookup in DB).
            threshold: Cosine similarity threshold (0-1).

        Returns:
            List of image file paths that match.
        """
        name = name_query.lower().strip()
        if name not in self.known_faces:
            logger.warning("Face for '%s' not found in database", name)
            return []

        target_embedding = np.array(self.known_faces[name])
        image_paths = self._get_image_paths()

        logger.info("Searching for '%s' across %d images", name, len(image_paths))

        results = []
        for img_path in image_paths:
            try:
                embeddings = DeepFace.represent(
                    img_path=img_path, model_name="Facenet", enforce_detection=False
                )
                for embedding_data in embeddings:
                    current_embedding = np.array(embedding_data["embedding"])
                    similarity = self._cosine_similarity(
                        target_embedding, current_embedding
                    )
                    if similarity > threshold:
                        results.append(img_path)
                        logger.debug(
                            "Face match found in %s (sim: %.3f)", img_path, similarity
                        )
                        break
            except Exception:
                continue

        return results

    # Internal

    def _get_image_paths(self) -> List[str]:
        """Return a cached list of image paths, re-scanning if stale."""
        now = time.time()
        if now - self._cache_timestamp > self.CACHE_TTL:
            include_dirs = config.include_directories
            exclude_dirs = config.exclude_directories
            paths, _ = fast_scan_for_media(include_dirs, exclude_dirs)
            self._cached_paths = [
                p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            self._cache_timestamp = now
        return self._cached_paths

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
