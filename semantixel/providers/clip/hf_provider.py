"""Hugging Face Transformers CLIP provider for image and text embeddings."""

import torch
import warnings
from typing import List, Union, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from semantixel.providers.base import CLIPProvider
from semantixel.providers.registry import provider
from semantixel.core.logging import logger
from semantixel.core.device import detect_device, unwrap_output, clear_gpu_cache

warnings.filterwarnings("ignore")


@provider("clip", "HF_transformers")
class HFCLIPProvider(CLIPProvider):
    """Hugging Face Transformers implementation of CLIP.

    Uses ``transformers.CLIPModel`` for zero-shot image classification
    and cross-modal retrieval.  Supports CPU, CUDA, and Apple Silicon (MPS).
    """

    def __init__(self, checkpoint: str = "openai/clip-vit-base-patch32"):
        self.checkpoint = checkpoint
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = detect_device()

    def load(self):
        """Load the CLIP model and processor onto the selected device."""
        if self.model is not None:
            return

        logger.info("Loading HF CLIP model: %s on %s", self.checkpoint, self.device)

        try:
            self.model = CLIPModel.from_pretrained(self.checkpoint, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        except (OSError, ValueError):
            logger.info("Model %s not found locally. Downloading...", self.checkpoint)
            self.model = CLIPModel.from_pretrained(self.checkpoint)
            self.processor = CLIPProcessor.from_pretrained(self.checkpoint)

        self.model.to(self.device)
        self.model.eval()
        clear_gpu_cache(self.device)

    def unload(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            logger.info("Unloading CLIP model: %s", self.checkpoint)
            self.model = None
            self.processor = None
            clear_gpu_cache(self.device)

    @staticmethod
    def _open_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """Open an image from a path or return a PIL Image as-is."""
        if isinstance(image_input, Image.Image):
            return image_input
        return Image.open(image_input).convert("RGB")

    def get_image_embeddings(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        """Compute L2-normalised CLIP image embeddings.

        Args:
            images: List of image file paths or PIL Image objects.

        Returns:
            List of embedding vectors as Python ``float`` lists.
        """
        if not images:
            return []

        self.load()

        with ThreadPoolExecutor() as executor:
            pil_images = list(executor.map(self._open_image, images))

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            image_features = unwrap_output(outputs)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return [feat.tolist() for feat in image_features.cpu()]

    def get_text_embeddings(self, text: str) -> List[float]:
        """Compute L2-normalised CLIP text embedding for a single query.

        Args:
            text: The text query.

        Returns:
            An embedding vector as a Python ``float`` list.
        """
        self.load()
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            outputs = self.model.get_text_features(**inputs)
            text_features = unwrap_output(outputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu()[0].tolist()
