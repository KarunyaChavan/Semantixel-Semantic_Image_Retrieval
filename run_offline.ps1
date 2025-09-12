# PowerShell script to run Semantic Image Retrieval in offline mode
Write-Host "Starting Semantic Image Retrieval in Offline Mode..." -ForegroundColor Green

# Activate the conda environment
Write-Host "Activating conda environment: SemanticImageRetrieval_v2" -ForegroundColor Yellow
conda activate SemanticImageRetrieval_v2

# Check if activation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Environment activated successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to activate environment. Please make sure 'SemanticImageRetrieval_v2' environment exists." -ForegroundColor Red
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

# Run the project
Write-Host "Starting the application..." -ForegroundColor Green
Write-Host "Note: URL-based image search will still work online, but models will run offline." -ForegroundColor Cyan

try {
    python main.py
} catch {
    Write-Host "Error running the application: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If this is the first run, models may need to be downloaded. Try running with internet connection first." -ForegroundColor Yellow
}

Write-Host "Application finished." -ForegroundColor Green
