# ============================================================================
# Semantixel — Setup Script (Windows / PowerShell)
# ============================================================================
# Usage:  .\setup.ps1 [[-EnvName] <string>]
#
# Default env_name: semantixel
#
# Steps performed:
#   1. Create conda environment with Python 3.11
#   2. Install PyTorch with GPU (CUDA 12.1) support via conda
#   3. Install remaining Python dependencies via pip
#   4. Generate default config.yaml if missing
# ============================================================================
param(
    [string]$EnvName = "semantixel"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Creating conda environment '$EnvName' with Python 3.11..." -ForegroundColor Cyan
conda create -y -n $EnvName python=3.11
if (-not $?) { exit 1 }

Write-Host "==> Activating environment..." -ForegroundColor Cyan
conda activate $EnvName
if (-not $?) { exit 1 }

Write-Host "==> Installing PyTorch with CUDA 12.4 support via pip..." -ForegroundColor Cyan
Write-Host "    (For CPU-only, use: --index-url https://download.pytorch.org/whl/cpu)" -ForegroundColor Yellow
Write-Host "    (For CUDA 11.8, use: --index-url https://download.pytorch.org/whl/cu118)" -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if (-not $?) { exit 1 }

Write-Host "==> Installing project dependencies..." -ForegroundColor Cyan
python -m pip install -e .
if (-not $?) { exit 1 }

Write-Host "==> Creating default config.yaml if missing..." -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath "config.yaml")) {
    $yaml = @"
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
"@
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath ".").Path + "/config.yaml", $yaml)
    Write-Host "    -> config.yaml created." -ForegroundColor Green
} else {
    Write-Host "    -> config.yaml already exists." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Semantixel setup complete!" -ForegroundColor Green
Write-Host "  Activate:  conda activate $EnvName" -ForegroundColor Yellow
Write-Host "  Run:       python main.py" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
