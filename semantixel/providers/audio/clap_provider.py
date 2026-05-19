import torch
import librosa
from typing import List, Union
from transformers import ClapModel, ClapProcessor
from semantixel.providers.base import BaseModelProvider
from semantixel.core.logging import logger

DEFAULT_CLAP_CHECKPOINT = "laion/clap-htsat-unfused"

class HFAudioCLAPProvider(BaseModelProvider):
    def __init__(self, checkpoint: str = DEFAULT_CLAP_CHECKPOINT):
        super().__init__()
        self.checkpoint = checkpoint
        self.processor = None
        self.model = None
        self.is_loaded = False
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def load(self):
        try:
            logger.info(f"Loading CLAP model into VRAM/RAM: {self.checkpoint} on {self.device}")
            self.processor = ClapProcessor.from_pretrained(self.checkpoint)
            self.model = ClapModel.from_pretrained(self.checkpoint).to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Successfully loaded CLAP model: {self.checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load CLAP model {self.checkpoint}: {e}")
            raise e

    def get_audio_embeddings(self, audio_path: str) -> list:
        if not self.is_loaded:
            self.load()
            
        try:
            y, sr = librosa.load(audio_path, sr=48000, duration=10.0)
            inputs = self.processor(audios=y, sampling_rate=48000, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
            
            # Normalize vector for cosine similarity math
            embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding[0].cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error getting CLAP audio embedding for {audio_path}: {e}")
            # CLAP dim is 512, return empty vector to shield ChromaDB
            return [0.0] * 512

    def get_text_embeddings(self, text: str) -> list:
        if not self.is_loaded:
            self.load()
            
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                
            embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error getting CLAP text embedding for '{text}': {e}")
            return [0.0] * 512

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.is_loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Unloaded CLAP model: {self.checkpoint}")
