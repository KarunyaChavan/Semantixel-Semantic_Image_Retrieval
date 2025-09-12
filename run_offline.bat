@echo off
echo Starting Semantic Image Retrieval in Offline Mode...

REM Activate the conda environment
echo Activating conda environment: SemanticImageRetrieval_v2
call conda activate SemanticImageRetrieval_v2

if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate environment. Please make sure 'SemanticImageRetrieval_v2' environment exists.
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

REM Run the project
echo Starting the application...
echo Note: URL-based image search will still work online, but models will run offline.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo Error running the application.
    echo If this is the first run, models may need to be downloaded. Try running with internet connection first.
    pause
    exit /b 1
)

echo Application finished.
pause
