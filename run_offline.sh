#!/bin/bash
# Bash script to run Semantic Image Retrieval in offline mode

echo "Starting Semantic Image Retrieval in Offline Mode..."

# Activate the conda environment
echo "Activating conda environment: Semantixel"

# Initialize conda if needed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" != "Semantixel" ]]; then
    # Try to activate the environment
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate Semantixel 2>/dev/null
    
    if [[ $? -ne 0 ]]; then
        echo "Failed to activate environment. Please make sure 'Semantixel' environment exists."
        echo "Create it with: conda create -n Semantixel python=3.11"
        exit 1
    fi
fi

echo "✓ Environment activated successfully!"

# Set offline environment variables and suppress warnings
echo "Setting offline mode environment variables..."
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY=False

# Check if Rust module is available
echo "Verifying Rust optimization module..."
python -c "import semantixel_scanner; print('✓ Rust module available - using optimized backend')" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "⚠ Warning: Rust module not found. Falling back to Python backend."
    echo "To build Rust module, run: cd semantixel_rust && maturin develop --release"
fi

# Run the project
echo "Starting the application..."
echo "Using Rust-accelerated directory scanning and CSV operations."
echo "URL-based image search will work online, models run in offline mode."
echo ""

python main.py

if [[ $? -ne 0 ]]; then
    echo ""
    echo "Error running the application."
    echo "If this is the first run, models may need to be downloaded. Try running with internet connection first."
    exit 1
fi

echo "✓ Application finished."
