import torch
import re
import numpy as np
import cv2
from typing import List, Union, Optional
from PIL import Image, ImageEnhance, ImageFilter
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor
from semantixel.providers.base import OCRProvider
from semantixel.core.logging import logger

class DoctrOCRProvider(OCRProvider):
    """
    Doctr implementation of OCR.
    """
    def __init__(self, det_arch: str = "db_mobilenet_v3_large", reco_arch: str = "crnn_mobilenet_v3_large"):
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.model = None
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def load(self):
        if self.model is not None:
            return
            
        logger.info(f"Loading Doctr OCR model: {self.det_arch}, {self.reco_arch} on {self.device}")
        self.model = ocr_predictor(self.det_arch, self.reco_arch, pretrained=True)
        self.model.to(self.device)

    def unload(self):
        if self.model is not None:
            logger.info("Unloading OCR model")
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _enhance_image(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        # Step 1: Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Step 2: Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # Step 3: Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # OpenCV ops
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_cv = cv2.bilateralFilter(image_cv, 9, 75, 75)
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        return image

    def _clean_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s\.\,\-\:\;\!\?]', '', text)
        words = text.split()
        words = [w for w in words if len(w) > 1 or w.isdigit()]
        text = ' '.join(words)
        return text if text else None

    def _process_page(self, page, threshold: float) -> Optional[str]:
        try:
            text = " ".join(
                word.value
                for block in page.blocks
                for line in block.lines
                for word in line.words
                if word.confidence > threshold
            )
        except Exception:
            text = None
        
        text = self._clean_text(text)
        
        if text is None:
            return None
        
        if text == "" or (
            not any(char.isalpha() for char in text) or len(text) < 3
            or all(len(word) == 1 for word in text.split() if word.isalpha())
        ):
            return None
            
        return text

    def apply_ocr(self, images: List[Union[str, Image.Image]], threshold: float = 0.4) -> List[Optional[str]]:
        if not images:
            return []
            
        self.load()
        
        with ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(self._enhance_image, images))
            
        output = self.model(processed_images)
        
        # Process pages
        texts = [self._process_page(p, threshold) for p in output.pages]
        return texts
