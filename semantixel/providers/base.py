from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
from PIL import Image

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
    @abstractmethod
    def transcribe(self, file_path: str) -> Optional[str]:
        """
        Transcribe an audio file into text.

        Args:
            file_path: Path to the input audio file.

        Returns:
            The transcribed text if successful, otherwise None.
        """
        pass