# Deployment and Usage

This document outlines the setup, execution, and deployment strategies for Semantixel.

## Requirements

- Python 3.11
- `requirements.txt` containing precise dependencies, including PyTorch, Transformers, and Flask.

## Local setup

1. Create a virtual environment and install all requisite packages.
    ```bash
    conda create -n semantixel python=3.11 -y
    conda activate semantixel
    pip install -r requirements.txt
    ```

2. Configure the application via the settings interface, which manages `config.yaml`:
    ```bash
    python settings.py
    ```

3. Launch the server and processing queues:
    ```bash
    python main.py
    ```

## Advanced Deployment

- Containerization: Semantixel is suitable for containerized deployment. A `Dockerfile` should install dependencies and expose the WSGI application defined in `wsgi.py`.
- Hardware Passthrough: When utilizing Docker, ensure that GPU resources are passed through using the `--gpus` flag for accelerated model inference.
- Cloud Hosting: The API layer can be securely deployed to cloud instances, while the static frontend can be served via standard web servers or CDNs.

Note: In production environments, it is recommended to run the Flask application behind a production-grade WSGI server such as Gunicorn or Waitress.
