import torch
from typing import Optional, Union, List
from transformers import pipeline
import transformers
from semantixel.providers.base import AudioProvider
from semantixel.core.logging import logger

transformers.logging.set_verbosity_error()

class HFAudioProvider(AudioProvider):
    """
    Hugging Face Transformers implementation of Audio Transcriptions (Whisper).
    """
    def __init__(self, checkpoint: str = "openai/whisper-tiny"):
        self.checkpoint = checkpoint
        self.pipe = None
        self.device_mapped = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        
    def load(self):
        if self.pipe is not None:
            return
            
        logger.info(f"Loading HF Audio model (Whisper): {self.checkpoint} on {self.device_mapped}")
        try:
            # Try to load from cache/local files first
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model=self.checkpoint, 
                device=self.device_mapped,
                model_kwargs={"local_files_only": True}
            )
        except Exception as e:
            logger.info(f"Audio model {self.checkpoint} not found locally or failed. Downloading... ({e})")
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model=self.checkpoint, 
                device=self.device_mapped
            )

    def unload(self):
        if self.pipe is not None:
            logger.info("Unloading Audio model")
            self.pipe = None
            if "cuda" in self.device_mapped:
                torch.cuda.empty_cache()

    def transcribe(self, file_path: str) -> Optional[str]:
        """
        Transcribes the provided audio file path.
        """
        self.load()
        try:
            import librosa
            # Whisper natively requires 16000Hz sampling rate.
            # We restrict to exactly the first 20 seconds to get semantic context instantly!
            y, sr = librosa.load(file_path, sr=16000, duration=20.0)
            
            audio_input = {"raw": y, "sampling_rate": sr}
            result = self.pipe(audio_input, return_timestamps=True, chunk_length_s=30)
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Error transcribing audio file {file_path}: {e}")
            return None
