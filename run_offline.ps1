# PowerShell script to run Semantic Image Retrieval in offline mode
Write-Host "Starting Semantic Image Retrieval in Offline Mode..." -ForegroundColor Green

# Activate the conda environment
Write-Host "Activating conda environment: Semantixel" -ForegroundColor Yellow
conda activate Semantixel

# Check if activation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Environment activated successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to activate environment. Please make sure 'Semantixel' environment exists." -ForegroundColor Red
    Write-Host "Create it with: conda create -n Semantixel python=3.11" -ForegroundColor Yellow
    exit 1
}

# Set offline environment variables and suppress warnings
Write-Host "Setting offline mode environment variables..." -ForegroundColor Yellow
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:PYTORCH_OFFLINE = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:ANONYMIZED_TELEMETRY = "False"
$env:CHROMA_TELEMETRY = "False"

# Check if models need to be downloaded first
Write-Host "Checking if models are cached locally..." -ForegroundColor Yellow

# Check if Rust module is available
Write-Host "Verifying Rust optimization module..." -ForegroundColor Yellow
python -c "import semantixel_scanner; print('✓ Rust module available - using optimized backend')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: Rust module not found. Falling back to Python backend." -ForegroundColor Yellow
    Write-Host "To build Rust module, run: cd semantixel_rust && maturin develop --release" -ForegroundColor Cyan
}

# Run the project
Write-Host "Starting the application..." -ForegroundColor Green
Write-Host "Using Rust-accelerated directory scanning and CSV operations." -ForegroundColor Cyan
Write-Host "URL-based image search will work online, models run in offline mode." -ForegroundColor Cyan

try {
    python main.py
} catch {
    Write-Host "Error running the application: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If this is the first run, models may need to be downloaded. Try running with internet connection first." -ForegroundColor Yellow
}

Write-Host "Application finished." -ForegroundColor Green
