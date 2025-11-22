@echo off
echo Starting Semantic Image Retrieval in Offline Mode...

REM Activate the conda environment
echo Activating conda environment: Semantixel
call conda activate Semantixel

if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate environment. Please make sure 'Semantixel' environment exists.
    echo Create it with: conda create -n Semantixel python=3.11
    pause
    exit /b 1
)

echo Environment activated successfully!

REM Set offline environment variables and suppress warnings
echo Setting offline mode environment variables...
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
set PYTORCH_OFFLINE=1
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
set ANONYMIZED_TELEMETRY=False
set CHROMA_TELEMETRY=False

REM Check if Rust module is available
echo Verifying Rust optimization module...
python -c "import semantixel_scanner; print('Rust module available - using optimized backend')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Rust module not found. Falling back to Python backend.
    echo To build Rust module, run: cd semantixel_rust ^&^& maturin develop --release
)

REM Run the project
echo Starting the application...
echo Using Rust-accelerated directory scanning and CSV operations.
echo URL-based image search will work online, models run in offline mode.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo Error running the application.
    echo If this is the first run, models may need to be downloaded. Try running with internet connection first.
    pause
    exit /b 1
)

echo Application finished.
pause
