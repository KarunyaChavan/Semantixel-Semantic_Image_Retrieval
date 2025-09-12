from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import yaml
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure TensorFlow to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow info and warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations messages

# Configure for offline usage
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
checkpoint = config["clip"]["HF_transformers_clip"]

# load model and tokenizer with local_files_only=True for offline usage
try:
    model = CLIPModel.from_pretrained(checkpoint, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
except OSError:
    print(f"Models not found locally. Downloading {checkpoint} (this only needs to happen once)...")
    # Download models if not cached
    model = CLIPModel.from_pretrained(checkpoint)
    processor = CLIPProcessor.from_pretrained(checkpoint, use_fast=True)
    print(f"Models downloaded successfully. Future runs will work offline.")
# move model to cuda if available
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model.to(device)

# Print device information for debugging
print(f"CLIP Model running on: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)} GB")
    # Enable memory efficiency for smaller GPUs
    torch.cuda.empty_cache()
    if torch.cuda.get_device_properties(0).total_memory < 6e9:  # Less than 6GB
        print("   Optimizing for low GPU memory...")


def open_image(image_path):
    """
    Opens an image from the specified file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image: An Image object representing the opened image.
    """
    image = Image.open(image_path)
    return image


def get_clip_image(image_paths):
    """
    Computes the image embeddings for a batch of images and calculates the average pixel value of the first channel of each image.

    This function opens each image from the specified paths, preprocesses them for the model, and computes their embeddings. Additionally, it calculates the average pixel value of the first channel as a simple feature for each image.

    Args:
        image_paths (list): List of image paths with length equal to batch_size

    Returns:
        list: A list containing the image embeddings for each image in the batch.
    """
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(open_image, image_paths))
    processed_image = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**processed_image)
    return image_features.cpu().squeeze(0).numpy().tolist()


def get_clip_text(text):
    """
    Gets the text embeddings for the given text.

    Args:
        text (str): The text to get embeddings for.

    Returns:
        list: The text embeddings as a list.
    """
    with torch.no_grad():
        processed_text = processor(text=text, return_tensors="pt").to(device)
        text_features = model.get_text_features(**processed_text)
    return text_features.cpu().squeeze(0).numpy().tolist()
