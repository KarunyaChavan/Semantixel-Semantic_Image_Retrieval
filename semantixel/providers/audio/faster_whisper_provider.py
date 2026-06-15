import torch
from typing import Optional
from semantixel.providers.base import AudioProvider
from semantixel.core.logging import logger, log_exception
from semantixel.utils import has_audio_stream

DEFAULT_WHISPER_CHECKPOINT = "tiny.en"

class FasterWhisperProvider(AudioProvider):
    """
    Faster-Whisper (CTranslate2) implementation of Audio Transcriptions.
    """
    def __init__(self, checkpoint: str = DEFAULT_WHISPER_CHECKPOINT):
        self.checkpoint = checkpoint
        self.model = None
        # Faster-Whisper requires explicit "cuda" or "cpu" strings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Determine compute type. float16 requires CUDA capability, fallback to int8.
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
    def load(self):
        if self.model is not None:
            return
            
        logger.info(f"Loading Faster Whisper model: {self.checkpoint} on {self.device} ({self.compute_type})")
        try:
            from faster_whisper import WhisperModel
            # Native download happens automatically if not cached
            self.model = WhisperModel(self.checkpoint, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper model: {e}")
            raise e

    def unload(self):
        if self.model is not None:
            logger.info("Unloading Faster Whisper model")
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def transcribe(self, file_path: str, max_duration: float = 60.0) -> Optional[str]:
        """
        Transcribes the provided audio file path natively.
        """
        self.load()
        try:
            import librosa
            if not has_audio_stream(file_path):
                logger.debug(f"No audio stream found in {file_path}, skipping transcription")
                return None
            duration = None if max_duration <= 0 else max_duration
            y, sr = librosa.load(file_path, sr=16000, duration=duration)

            segments, info = self.model.transcribe(y, beam_size=5)

            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            if "cublas" in str(e).lower() and self.device == "cuda":
                logger.warning(f"CUDA transcription failed: {e}. Retrying on CPU fallback.")
                self.unload()
                self.device = "cpu"
                self.compute_type = "int8"
                return self.transcribe(file_path, max_duration)

            log_exception(logger, f"Error transcribing {file_path} via Faster Whisper")
            return None
