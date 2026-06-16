"""Hugging Face Transformers text embedding provider (sentence-transformers)."""

import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from semantixel.providers.base import TextEmbeddingProvider
from semantixel.providers.registry import provider
from semantixel.core.logging import logger
from semantixel.core.device import detect_device, clear_gpu_cache


@provider("text", "HF_transformers")
class HFTextEmbeddingProvider(TextEmbeddingProvider):
    """Hugging Face Transformers implementation of dense text embeddings.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` (or a user-specified
    checkpoint) with mean-pooling and L2 normalisation.
    """

    def __init__(self, checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.checkpoint = checkpoint
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.device = detect_device()

    def load(self):
        """Load the tokenizer and model onto the selected device."""
        if self.model is not None:
            return

        logger.info("Loading HF Text Embedding model: %s on %s", self.checkpoint, self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.checkpoint, local_files_only=True)
        except (OSError, ValueError):
            logger.info("Model %s not found locally. Downloading...", self.checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModel.from_pretrained(self.checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def unload(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            logger.info("Unloading Text Embedding model")
            self.model = None
            self.tokenizer = None
            clear_gpu_cache(self.device)

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Mean-pool token embeddings weighted by attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, text: str) -> List[float]:
        """Compute a mean-pooled, L2-normalised embedding for *text*.

        Args:
            text: Input text to embed.

        Returns:
            Dense embedding vector as a Python ``float`` list.
        """
        self.load()

        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()
