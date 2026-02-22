import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from PIL import Image
import os
import yaml
import warnings

try:
    import mobileclip
except ImportError:
    import subprocess

    subprocess.run(
        [
            "pip",
            "install",
            "git+https://github.com/apple/ml-mobileclip.git@main#egg=mobileclip",
            "--no-deps",
        ]
    )
    import mobileclip


warnings.filterwarnings("ignore")

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
checkpoint = config["clip"]["mobileclip_checkpoint"]


def download_mobile_clip(checkpoint):
    """
    Downloads the MobileCLIP checkpoint file.

    Args:
        checkpoint (str): The name of the checkpoint file.
    """
    url = f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{checkpoint}.pt"
    response = requests.get(url, stream=True)

    file_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024  # 1 KB
    filename = url.split("/")[-1]

    progress = tqdm(
        response.iter_content(chunk_size),
        f"Downloading {filename}",
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    # create checkpoints path if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/{filename}", "wb") as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))


if not os.path.exists(f"checkpoints/{checkpoint}.pt"):
    print(f"checkpoints/{checkpoint}.pt")
    download_mobile_clip(checkpoint)

# load model and tokenizer
model, _, preprocess = mobileclip.create_model_and_transforms(
    checkpoint, pretrained=f"checkpoints/{checkpoint}.pt"
)
tokenizer = mobileclip.get_tokenizer(checkpoint)
# move model to cuda if available
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

model.to(device)


def preprocess_image(image_input):
    """
    Preprocesses a single image or image path.

    Args:
        image_input (str or PIL.Image): The path to the image file or a PIL Image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        # Assume it's already a PIL Image
        image = image_input
    return preprocess(image.convert("RGB"))


def get_clip_image(image_input):
    """
    Computes the image embeddings for a batch of images.

    This function accepts either a list of file paths or a list of PIL Image objects, 
    preprocesses them for the model, and computes their embeddings.

    Args:
        image_input (list): List of image paths or PIL Image objects with length equal to batch_size

    Returns:
        list: A list containing the image embeddings for each image in the batch.
    """
    if not image_input:
        return []
        
    # Use ThreadPoolExecutor to preprocess images in parallel
    with ThreadPoolExecutor() as executor:
        preprocessed_images = list(executor.map(preprocess_image, image_input))
        
    # Stack the preprocessed images into a tensor
    images_tensor = torch.stack(preprocessed_images).to(device)

    # Encode the batch of images
    with torch.no_grad():
        image_features_batch = model.encode_image(images_tensor)

    # Convert the batch of image features to a list of lists
    image_features_list = image_features_batch.cpu().squeeze(1).numpy().tolist()

    return image_features_list


def get_clip_text(text):
    """
    Gets the text embeddings for the given text.

    Args:
        text (str): The text to get embeddings for.

    Returns:
        list: The text embeddings as a list.
    """
    text = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.cpu().squeeze(0).numpy().tolist()
