from typing import Optional
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.providers.clip.hf_provider import HFCLIPProvider
from semantixel.providers.ocr.doctr_provider import DoctrOCRProvider
from semantixel.providers.text.hf_provider import HFTextEmbeddingProvider
from semantixel.providers.audio.hf_audio_provider import HFAudioProvider
from semantixel.providers.audio.faster_whisper_provider import FasterWhisperProvider
from semantixel.providers.audio.clap_provider import HFAudioCLAPProvider

class ModelManager:
    """
    Singleton manager for all AI models.
    Provides lazy loading and centralized access to model providers.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
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

    @property
    def clip(self):
        if self._clip_provider is None:
            provider_type = config.clip.provider
            if provider_type == "HF_transformers":
                self._clip_provider = HFCLIPProvider(checkpoint=config.clip.HF_transformers_clip)
            else:
                # Fallback or other providers like MobileCLIP
                logger.warning(f"Unsupported CLIP provider: {provider_type}. Falling back to HF.")
                self._clip_provider = HFCLIPProvider()
        return self._clip_provider

    @property
    def ocr(self):
        if self._ocr_provider is None:
            provider_type = config.ocr_provider
            if provider_type == "doctr":
                self._ocr_provider = DoctrOCRProvider()
            else:
                logger.warning(f"Unsupported OCR provider: {provider_type}. Falling back to Doctr.")
                self._ocr_provider = DoctrOCRProvider()
        return self._ocr_provider

    @property
    def text_embed(self):
        if self._text_provider is None:
            provider_type = config.text_embed.provider
            if provider_type == "HF_transformers":
                self._text_provider = HFTextEmbeddingProvider(checkpoint=config.text_embed.HF_transformers_embeddings)
            else:
                logger.warning(f"Unsupported Text Embedding provider: {provider_type}. Falling back to HF.")
                self._text_provider = HFTextEmbeddingProvider()
        return self._text_provider

    @property
    def audio(self):
        if self._audio_provider is None:
            provider_type = config.audio.provider
            if provider_type == "faster_whisper":
                self._audio_provider = FasterWhisperProvider(checkpoint=config.audio.faster_whisper_model)
            elif provider_type == "HF_transformers":
                self._audio_provider = HFAudioProvider(checkpoint=config.audio.HF_transformers_whisper)
            else:
                logger.warning(f"Unsupported Audio provider: {provider_type}. Falling back to faster_whisper.")
                self._audio_provider = FasterWhisperProvider()
        return self._audio_provider

    @property
    def clap(self):
        if self._clap_provider is None:
            self._clap_provider = HFAudioCLAPProvider()
        return self._clap_provider

    def unload_all(self):
        """Unload all models to free memory/VRAM."""
        if self._clip_provider:
            self._clip_provider.unload()
        if self._ocr_provider:
            self._ocr_provider.unload()
        if self._text_provider:
            self._text_provider.unload()
        if self._audio_provider:
            self._audio_provider.unload()
        if self._clap_provider:
            self._clap_provider.unload()

# Global model manager instance
model_manager = ModelManager()
