import subprocess
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
from PIL import Image
from semantixel.core.logging import logger

class BaseModelProvider(ABC):
    """
    Abstract base class for all model providers (CLIP, OCR, Text Embeddings, Audio).
    Ensures a consistent interface across different implementations.
    """
    
    @abstractmethod
    def load(self):
        """Load the model into memory/GPU."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model to free up resources."""
        pass

class CLIPProvider(BaseModelProvider):
    @abstractmethod
    def get_image_embeddings(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        pass

    @abstractmethod
    def get_text_embeddings(self, text: str) -> List[float]:
        pass

class OCRProvider(BaseModelProvider):
    @abstractmethod
    def apply_ocr(self, images: List[Union[str, Image.Image]], threshold: float = 0.4) -> List[Optional[str]]:
        pass

class TextEmbeddingProvider(BaseModelProvider):
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        pass

class AudioProvider(BaseModelProvider):
    @staticmethod
    def _has_audio_stream(file_path: str) -> bool:
        """Check whether *file_path* contains an audio track.

        Delegates to ``ffprobe``. Returns ``False`` when no audio stream
        is found so callers can skip processing. On probe failure the
        method is permissive (returns ``True``) to avoid silently
        dropping files when the probe tool is missing.

        Args:
            file_path: Absolute path to the media file.

        Returns:
            ``True`` if an audio stream is present (or probe failed).
        """
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", file_path],
                capture_output=True, text=True, timeout=30
            )
            return "audio" in result.stdout
        except Exception:
            return True

    @abstractmethod
    def transcribe(self, file_path: str, max_duration: float = 60.0) -> Optional[str]:
        """
        Transcribe an audio file into text.

        Args:
            file_path: Path to the input audio file.
            max_duration: Maximum seconds of audio to transcribe (0 = full file).

        Returns:
            The transcribed text if successful, otherwise None.
        """
        pass