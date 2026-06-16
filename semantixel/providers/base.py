"""Abstract base classes for all model providers.

Each modality (CLIP, OCR, text embeddings, audio) defines a dedicated
interface that all implementations must satisfy.  New providers should
inherit from these classes and register themselves via the
:mod:`semantixel.providers.registry` decorator.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from PIL import Image
class BaseModelProvider(ABC):
    """Minimal lifecycle contract for any ML model provider.

    All providers must implement :meth:`load` and :meth:`unload` to
    manage model memory.  The :mod:`ModelManager` guarantees these are
    called appropriately.
    """

    @abstractmethod
    def load(self):
        """Load the model into memory / GPU."""

    @abstractmethod
    def unload(self):
        """Unload the model and free resources."""


class CLIPProvider(BaseModelProvider):
    """Interface for CLIP-based cross-modal embedding providers."""

    @abstractmethod
    def get_image_embeddings(
        self, images: List[Union[str, Image.Image]]
    ) -> List[List[float]]:
        """Embed one or more images into a shared latent space.

        Args:
            images: Paths or PIL Images.

        Returns:
            List of L2-normalised embedding vectors.
        """

    @abstractmethod
    def get_text_embeddings(self, text: str) -> List[float]:
        """Embed a text query into the same latent space as images.

        Args:
            text: Query string.

        Returns:
            L2-normalised embedding vector.
        """


class OCRProvider(BaseModelProvider):
    """Interface for OCR / text-extraction providers."""

    @abstractmethod
    def apply_ocr(
        self, images: List[Union[str, Image.Image]], threshold: float = 0.4
    ) -> List[Optional[str]]:
        """Extract text from a batch of images.

        Args:
            images: Paths or PIL Images.
            threshold: Minimum per-word confidence (0-1).

        Returns:
            Extracted text for each image, or ``None`` for empty results.
        """


class TextEmbeddingProvider(BaseModelProvider):
    """Interface for dense text embedding providers (e.g. sentence-transformers)."""

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            Dense embedding vector.
        """


class AudioProvider(BaseModelProvider):
    """Interface for audio transcription providers (e.g. Whisper)."""

    @abstractmethod
    def transcribe(
        self, file_path: str, max_duration: float = 60.0
    ) -> Optional[str]:
        """Transcribe an audio file to text.

        Args:
            file_path: Path to the audio file.
            max_duration: Maximum seconds to process (0 = full file).

        Returns:
            Transcribed text, or ``None`` on failure.
        """
