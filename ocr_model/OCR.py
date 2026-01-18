import torch
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import re

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model = ocr_predictor(
    "db_mobilenet_v3_large", "crnn_mobilenet_v3_large", pretrained=True
)
model.to(device)


def enhance_image_for_ocr(image_path):
    """
    Preprocesses image for better OCR detection by enhancing contrast,
    reducing noise, and optimizing brightness.

    Args:
        image_path (str): The path to the image file.
    Returns:
        np.array: The enhanced preprocessed image tensor.
    """
    # Open image
    image = Image.open(image_path).convert('RGB')
    
    # Step 1: Enhance contrast to make text stand out
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    # Step 2: Enhance brightness for very dark images
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)  # Slight brightness boost
    
    # Step 3: Apply sharpening to enhance text edges
    image = image.filter(ImageFilter.SHARPEN)
    
    # Convert to numpy array for OpenCV operations
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Step 4: Apply bilateral filtering to denoise while preserving edges
    image_cv = cv2.bilateralFilter(image_cv, 9, 75, 75)
    
    # Step 5: Convert back to RGB for doctr
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Ensure proper format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    return image


def process_image(image_path):
    """
    Opens and preprocesses a single image with enhancement for better OCR.

    Args:
        image_path (str): The path to the image file.
    Returns:
        np.array: The enhanced preprocessed image tensor.
    """
    return enhance_image_for_ocr(image_path)


def clean_text(text):
    """
    Cleans and normalizes OCR output text.
    
    Args:
        text (str): Raw OCR text output.
    Returns:
        str: Cleaned and normalized text.
    """
    if not text:
        return None
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\.\,\-\:\;\!\?]', '', text)
    
    # Remove standalone single characters
    words = text.split()
    words = [w for w in words if len(w) > 1 or w.isdigit()]
    text = ' '.join(words)
    
    return text if text else None


def process_page(page, OCR_threshold):
    """
    Processes a single OCR page and extracts text based on a confidence threshold,
    with text cleaning and validation.

    Args:
        page: An OCR page object containing blocks, lines, and words.
        OCR_threshold (float): The confidence threshold for including words in the extracted text.

    Returns:
        str: The cleaned extracted text if it meets the criteria, otherwise None.
    """
    blocks = page.blocks
    try:
        text = " ".join(
            word.value
            for block in blocks
            for line in block.lines
            for word in line.words
            if word.confidence > OCR_threshold
        )
    except:
        text = None
    
    # Clean the extracted text
    text = clean_text(text)
    
    # Validate text quality
    if text is None:
        return None
    
    if text == "" or (
        text is not None
        and (not any(char.isalpha() for char in text) or len(text) < 3)
        or all(len(word) == 1 for word in text.split() if word.isalpha())
    ):
        return None
    
    return text


def apply_OCR(image_paths, OCR_threshold=0.4):
    """
    Applies Optical Character Recognition (OCR) on preprocessed images and returns recognized text.
    
    Images are enhanced for better OCR detection before processing.

    Args:
        image_paths (list of str): The paths to the image files.
        OCR_threshold (float, optional): The confidence threshold for OCR detection. Defaults to 0.4.
                                        Lower values capture more text but may include errors.
                                        Higher values (0.6+) are more conservative.

    Returns:
        list of str or None: The recognized and cleaned text for each image if any text is detected, 
                            otherwise None.
    """
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_paths))
    results = model(images)

    def process_page_wrapper(page):
        return process_page(page, OCR_threshold)

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(process_page_wrapper, results.pages))

    return texts
