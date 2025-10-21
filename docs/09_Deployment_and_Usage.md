# Deployment and Usage

This document explains how to set up and run Semantixel locally and options for deployment.

## Requirements

- Python 3.11
- `requirements.txt` for Python dependencies (present at repository root). Key libraries include torch, transformers, numpy, and pillow.

## Local setup

1. Create a virtual environment and install dependencies. Example (PowerShell):
    ```
    conda create -n Semantixel python=3.11 -y
    pip install -r requirements.txt
    ```

2. Run settings.py to configure models used and directories to be included and excluded from scanning:

    # Run Settings
    ``` 
    python settings.py
    ```

3. Run the application to open the Web UI/Flowlauncher Plugin:

    # Run server
    ```
    python main.py
    ```
    
## Docker and cloud

- Containerize: create a `Dockerfile` that installs Python and project dependencies, copies the code, and exposes the server port.
- GPU support: use an appropriate CUDA-enabled base image and the `--gpus` flag for `docker run`.
- Hosting: backend can be deployed to cloud VMs, Kubernetes, or serverless endpoints; the static UI can be hosted on GitHub Pages or any static host.

⚙️ Notes:
- If you plan to run indexing at scale, use a GPU-enabled instance for feature extraction.
- For production search, prefer FAISS or an external vector DB for performance and scalability.
