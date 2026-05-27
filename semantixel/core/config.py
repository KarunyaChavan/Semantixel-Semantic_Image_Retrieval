import os
import yaml
import shutil
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class CLIPConfig(BaseModel):
    """
    Configuration for CLIP-based embedding models.

    Attributes:
        HF_transformers_clip: Hugging Face CLIP model checkpoint.
        mobileclip_checkpoint: MobileCLIP model identifier.
        provider: Embedding provider backend.
    """
    HF_transformers_clip: str = "openai/clip-vit-base-patch32"
    mobileclip_checkpoint: str = "mobileclip_s0"
    provider: str = "HF_transformers"

class TextEmbedConfig(BaseModel):
    """
    Configuration for text embedding models.

    Attributes:
        HF_transformers_embeddings: Hugging Face text embedding model checkpoint.
        embedding_gguf: GGUF format embedding model path.
        ollama_embeddings: Ollama text embedding model name.
        openai_api_key: API key for OpenAI services.
        openai_endpoint: Endpoint for OpenAI services.
    """
    HF_transformers_embeddings: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_gguf: str = ""
    ollama_embeddings: str = ""
    openai_api_key: str = ""
    openai_endpoint: str = ""
    openai_model: str = ""
    provider: str = "HF_transformers"

class AudioConfig(BaseModel):
    """
    Configuration for audio transcription models.

    Attributes:
        HF_transformers_whisper: Hugging Face Whisper model checkpoint.
        faster_whisper_model: Faster Whisper model identifier.
        provider: Transcription provider backend.
    """
    HF_transformers_whisper: str = "openai/whisper-tiny"
    faster_whisper_model: str = "tiny.en"
    provider: str = "faster_whisper"
    
class GoogleDriveConfig(BaseModel):
    """
    Configuration for Google Drive integration.

    Attributes:
        enabled: Whether Google Drive integration is enabled.
        client_secret_file: Path to the client secret file.
        token_file: Path to the token file.
        redirect_uri: The redirect URI for the Google Drive API.
        folder_ids: A list of folder IDs to scan.
    """
    enabled: bool = False
    client_secret_file: str = ""
    token_file: str = "google_drive_token.json"
    redirect_uri: str = ""
    folder_ids: List[str] = Field(default_factory=list)
    include_shared_drives: bool = False
    page_size: int = 100
    image_mime_types: List[str] = Field(default_factory=lambda: [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    ])

class SemantixelConfig(BaseSettings):
    """
    Main configuration class for Semantixel, encompassing all settings for model providers, search parameters, and integrations.
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
        case_sensitive=False
    )

def load_config(config_path: str = "config.yaml", default_path: str = "config.default.yaml") -> SemantixelConfig:
    """
    Loads configuration from a YAML file, falling back to defaults if necessary.
    Automatically creates the config file from the default if it doesn't exist.
    """
    if not os.path.exists(config_path):
        if os.path.exists(default_path):
            print(f"Config file not found. Creating {config_path} from {default_path}")
            shutil.copy(default_path, config_path)
        else:
            # If even default is missing, we just return the default Pydantic model
            # and maybe save it as config.yaml?
            print(f"Warning: Neither {config_path} nor {default_path} found. Using system defaults.")
            return SemantixelConfig()

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    
    return SemantixelConfig(**config_data)

# Global config instance
config = load_config()
