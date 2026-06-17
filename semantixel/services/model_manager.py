"""Singleton model lifecycle manager.

Provides lazy initialisation and centralised access to all ML models.
Uses the :class:`ProviderRegistry` so that adding a new provider
implementation does **not** require modifying this file.
"""

from typing import Optional
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.providers.registry import ProviderRegistry, ProviderRegistryError


class ModelManager:
    """Singleton that holds lazy references to all model providers.

    Access each model via a read-only property (``.clip``, ``.ocr``,
    etc.).  The underlying provider is loaded on first access and cached
    for the lifetime of the process.

    Attributes:
        clip: CLIP image/text embedding provider.
        ocr: OCR text-extraction provider.
        text_embed: Dense text embedding provider.
        audio: Audio transcription provider.
        clap: CLAP audio/text embedding provider.
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._clip_provider = None
        self._ocr_provider = None
        self._text_provider = None
        self._audio_provider = None
        self._clap_provider = None
        self._initialized = True

    # Provider resolution lookup table
    # Map (config attribute, registry category, default name)
    # to avoid repetitive if/elif chains.

    _PROVIDER_MAP = {
        "clip": ("clip", "clip", "HF_transformers"),
        "ocr": ("ocr_provider", "ocr", "doctr"),
        "text_embed": ("text_embed", "text", "HF_transformers"),
    }

    @property
    def clip(self):
        """CLIP image/text embedding provider."""
        if self._clip_provider is None:
            self._clip_provider = self._resolve("clip", config.clip.provider)
        return self._clip_provider

    @property
    def ocr(self):
        """OCR text-extraction provider."""
        if self._ocr_provider is None:
            self._ocr_provider = self._resolve("ocr", config.ocr_provider)
        return self._ocr_provider

    @property
    def text_embed(self):
        """Dense text embedding provider."""
        if self._text_provider is None:
            self._text_provider = self._resolve("text", config.text_embed.provider)
        return self._text_provider

    @property
    def audio(self):
        """Audio transcription provider."""
        if self._audio_provider is None:
            self._audio_provider = self._resolve_audio()
        return self._audio_provider

    @property
    def clap(self):
        """CLAP audio/text embedding provider."""
        if self._clap_provider is None:
            self._clap_provider = self._resolve("clap", "HF_transformers")
        return self._clap_provider

    # Internal helpers

    @staticmethod
    def _resolve(category: str, name: str):
        """Instantiate a provider via the registry.

        Falls back to the default provider for the category if *name*
        is not found.
        """
        try:
            return ProviderRegistry.get(category, name)
        except ProviderRegistryError:
            logger.warning(
                "Provider '%s/%s' not found. Falling back to default.", category, name
            )
            defaults = {
                "clip": "HF_transformers",
                "ocr": "doctr",
                "text": "HF_transformers",
                "clap": "HF_transformers",
            }
            return ProviderRegistry.get(category, defaults.get(category, name))

    def _resolve_audio(self):
        """Resolve the audio provider, handling the dual-name config."""
        name = config.audio.provider
        try:
            return ProviderRegistry.get("audio", name)
        except ProviderRegistryError:
            logger.warning(
                "Audio provider '%s' not found. Falling back to faster_whisper.", name
            )
            return ProviderRegistry.get("audio", "faster_whisper")

    def unload_all(self):
        """Unload every provider and free GPU memory."""
        for attr in (
            "_clip_provider",
            "_ocr_provider",
            "_text_provider",
            "_audio_provider",
            "_clap_provider",
        ):
            provider = getattr(self, attr, None)
            if provider is not None:
                try:
                    provider.unload()
                except Exception as exc:
                    logger.warning("Error unloading %s: %s", attr, exc)
                setattr(self, attr, None)


# Global singleton
model_manager = ModelManager()
