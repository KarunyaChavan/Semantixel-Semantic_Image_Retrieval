#!/usr/bin/env bash
# ============================================================================
# Semantixel — Setup Script (Unix / macOS / Linux)
# ============================================================================
# Usage:  bash setup.sh [env_name]
#
# Default env_name: semantixel
#
# Steps performed:
#   1. Create conda environment with Python 3.11
#   2. Install PyTorch with GPU (CUDA 12.1) support via conda
#   3. Install remaining Python dependencies via pip
#   4. Generate default config.yaml if missing
# ============================================================================
set -euo pipefail

ENV_NAME="${1:-semantixel}"
PYTHON_VERSION="3.11"

echo "==> Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "==> Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch with CUDA 12.4 support via pip..."
echo "    (For CPU-only, use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu)"
echo "    (For CUDA 11.8, use: --index-url https://download.pytorch.org/whl/cu118)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing project dependencies..."
pip install -e .

echo "==> Creating default config.yaml if missing..."
if [ ! -f config.yaml ]; then
    cat > config.yaml << 'YAML'
port: 23107
batch_size: 16
deep_scan: true
ocr_provider: "doctr"
scan_method: "default"
exclude_directories: []
include_directories: []

clip:
  HF_transformers_clip: "openai/clip-vit-base-patch32"
  mobileclip_checkpoint: "mobileclip_s0"
  provider: "HF_transformers"

text_embed:
  HF_transformers_embeddings: "sentence-transformers/all-MiniLM-L6-v2"
  provider: "HF_transformers"

audio:
  enabled: true
  transcription_enabled: true
  clap_enabled: true
  max_duration_seconds: 0
  HF_transformers_whisper: "openai/whisper-tiny"
  faster_whisper_model: "tiny.en"
  provider: "faster_whisper"
  transcription_max_duration: 60.0

google_drive:
  enabled: false
  token_file: "google_drive_token.json"
  folder_ids: []
  include_shared_drives: false
  page_size: 100
YAML
    echo "    -> config.yaml created."
else
    echo "    -> config.yaml already exists."
fi

echo ""
echo "============================================"
echo "  Semantixel setup complete!"
echo "  Activate:  conda activate ${ENV_NAME}"
echo "  Run:       python main.py"
echo "============================================"
