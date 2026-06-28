"""Pydantic-based configuration management for Semantixel.

Uses ``yaml`` for the on-disk format and ``pydantic-settings`` for
validation.  A module-level :data:`config` singleton is created on
first import; call :func:`reload_config` to re-read the YAML file.
"""

import os
from functools import lru_cache
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CLIPConfig(BaseModel):
    """Settings for the CLIP image/text embedding provider.

    Attributes:
        HF_transformers_clip: Hugging Face model ID for CLIP.
        mobileclip_checkpoint: MobileCLI checkpoint name (reserved).
        provider: Active provider name (``"HF_transformers"``).
    """

    HF_transformers_clip: str = "openai/clip-vit-base-patch32"
    mobileclip_checkpoint: str = "mobileclip_s0"
    provider: str = "HF_transformers"


class TextEmbedConfig(BaseModel):
    """Settings for the text embedding provider.

    Attributes:
        HF_transformers_embeddings: Hugging Face model ID for dense embeddings.
        embedding_gguf: Path to a GGUF embedding model (optional).
        ollama_embeddings: Ollama model name (optional).
        openai_api_key: OpenAI API key for cloud embeddings (optional).
        openai_endpoint: Custom OpenAI-compatible endpoint (optional).
        openai_model: OpenAI model name (optional).
        provider: Active provider name (``"HF_transformers"``).
    """

    HF_transformers_embeddings: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_gguf: str = ""
    ollama_embeddings: str = ""
    openai_api_key: str = ""
    openai_endpoint: str = ""
    openai_model: str = ""
    provider: str = "HF_transformers"


class AudioConfig(BaseModel):
    """Settings for audio transcription and CLAP embedding.

    Attributes:
        enabled: Master switch for audio processing.
        transcription_enabled: Enable Whisper speech-to-text.
        clap_enabled: Enable CLAP ambient audio embeddings.
        max_duration_seconds: Skip files longer than this (0 = no limit).
        HF_transformers_whisper: Hugging Face model ID for Whisper.
        faster_whisper_model: Faster-Whisper model size (e.g. ``"tiny.en"``).
        provider: Active provider (``"faster_whisper"`` or ``"HF_transformers"``).
        transcription_max_duration: Max seconds of audio to transcribe at once.
    """

    enabled: bool = True
    transcription_enabled: bool = True
    clap_enabled: bool = True
    max_duration_seconds: float = 0
    HF_transformers_whisper: str = "openai/whisper-tiny"
    faster_whisper_model: str = "tiny.en"
    provider: str = "faster_whisper"
    transcription_max_duration: float = 60.0


class GoogleDriveConfig(BaseModel):
    """Settings for Google Drive integration.

    Attributes:
        enabled: Master switch.
        client_secret_file: Path to the OAuth client secrets JSON.
        token_file: Where to persist the OAuth token.
        redirect_uri: Custom redirect URI (optional).
        folder_ids: Restrict scanning to these folder IDs.
        include_shared_drives: Whether to include shared/team drives.
        page_size: Files per API page (max 1000).
        image_mime_types: MIME types accepted as images.
    """

    enabled: bool = False
    client_secret_file: str = ""
    token_file: str = "google_drive_token.json"
    redirect_uri: str = "http://localhost:23107/integrations/google_drive/auth/callback"
    folder_ids: List[str] = Field(default_factory=list)
    include_shared_drives: bool = False
    page_size: int = 100
    image_mime_types: List[str] = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
        ]
    )


class SemantixelConfig(BaseSettings):
    """Root configuration model for the Semantixel application.

    All fields can be overridden via environment variables with the
    ``SEMANTIXEL_`` prefix (e.g. ``SEMANTIXEL_PORT=8080``).

    Attributes:
        audio: Audio processing settings.
        batch_size: Number of items to process in a single model batch.
        clip: CLIP model settings.
        deep_scan: Re-index even if entry already exists.
        exclude_directories: Glob patterns / paths to skip during scan.
        google_drive: Google Drive integration settings.
        include_directories: Directories to include in the scan.
        ocr_provider: Active OCR provider name (``"doctr"``).
        port: Port for the Flask web server.
        scan_method: Scan strategy (reserved).
        text_embed: Text embedding settings.
    """

    audio: AudioConfig = Field(default_factory=AudioConfig)
    batch_size: int = 16
    clip: CLIPConfig = Field(default_factory=CLIPConfig)
    deep_scan: bool = True
    exclude_directories: List[str] = Field(default_factory=list)
    google_drive: GoogleDriveConfig = Field(default_factory=GoogleDriveConfig)
    include_directories: List[str] = Field(default_factory=list)
    ocr_provider: str = "doctr"
    port: int = 23107
    scan_method: str = "default"
    text_embed: TextEmbedConfig = Field(default_factory=TextEmbedConfig)

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIXEL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )


# Module-level singleton

_CONFIG_PATH: str = "config.yaml"
_DEFAULT_PATH: str = "config.default.yaml"


def load_config(
    config_path: str = _CONFIG_PATH, default_path: str = _DEFAULT_PATH
) -> SemantixelConfig:
    """Read and validate the YAML config file.

    Args:
        config_path: Path to the user's ``config.yaml``.
        default_path: Path to distribute defaults (optional).

    Returns:
        A validated :class:`SemantixelConfig` instance.
    """
    import shutil
    import yaml

    if not os.path.exists(config_path):
        if os.path.exists(default_path):
            print(
                "Config file not found. Creating %s from %s" % (config_path, default_path)
            )
            shutil.copy(default_path, config_path)
        else:
            print(
                "Warning: Neither %s nor %s found. Using system defaults."
                % (config_path, default_path)
            )
            return SemantixelConfig()

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config_data = yaml.safe_load(f) or {}

    # Auto-detect Google Drive settings from the environment.
    gd = config_data.get("google_drive")
    if not gd:
        gd = {}
        config_data["google_drive"] = gd

    # Auto-detect client secret file.
    if not gd.get("client_secret_file"):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        try:
            secrets = sorted(
                f for f in os.listdir(config_dir)
                if f.startswith("client_secret_") and f.endswith(".json")
            )
            if secrets:
                gd["client_secret_file"] = os.path.join(config_dir, secrets[0])
        except OSError:
            pass

    # Use the default redirect_uri if not explicitly set.
    if not gd.get("redirect_uri"):
        gd["redirect_uri"] = GoogleDriveConfig.model_fields["redirect_uri"].default

    return SemantixelConfig(**config_data)


@lru_cache(maxsize=1)
def _get_cached_config() -> SemantixelConfig:
    """Return the cached config singleton.

    Use :func:`reload_config` to invalidate the cache.
    """
    return load_config()


def reload_config() -> SemantixelConfig:
    """Re-read the YAML file and return a fresh :class:`SemantixelConfig`.

    Call this when the config file changes at runtime.
    """
    _get_cached_config.cache_clear()
    return _get_cached_config()


config: SemantixelConfig = _get_cached_config()
"""The global configuration singleton."""
