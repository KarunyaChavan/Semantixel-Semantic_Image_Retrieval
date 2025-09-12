import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
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
checkpoint = config["text_embed"]["HF_transformers_embeddings"]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Load models with offline configuration
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
except OSError:
    print(f"Models not found locally. Downloading bert-base-uncased and {checkpoint} (this only needs to happen once)...")
    # Download models if not cached
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    print(f"Models downloaded successfully. Future runs will work offline.")
model.eval()

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model = model.to(device)

# Print device information for debugging
print(f"Text Embeddings Model running on: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)} GB")


def get_text_embeddings(text, norm=False):
    """
    Gets the text embeddings for the given text.

    Args:
        text (str): The text to get embeddings for.

    Returns:
        list: The text embeddings as a list.
    """
    encoded_input = tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    if norm:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().squeeze(0).numpy().tolist()
