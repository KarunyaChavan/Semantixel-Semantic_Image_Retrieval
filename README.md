# Semantixel

Semantixel is an advanced semantic media retrieval system for local and connected image sources. It seamlessly indexes visual content using CLIP embeddings, extracts on-image text via OCR, and exposes robust search workflows for text-to-image, image-to-image, and OCR-backed retrieval through a lightweight web interface.

The project is tailored for personal knowledge bases, research datasets, screenshot archives, and comprehensive media collections where traditional keyword search is insufficient.

<p align="center">
  <a href="UI/Semantixel WebUI/assets/SemantiXel__AI_Image_Search.mp4">
    <img src="UI/Semantixel WebUI/assets/icon.png" width="200" alt="Watch Demo Video"/>
  </a>
  <br/>
  <strong><a href="UI/Semantixel WebUI/assets/SemantiXel__AI_Image_Search.mp4">Watch Demo Video</a></strong>
</p>

## Features

- **Natural-Language Image Retrieval**: Leverage CLIP text and image embeddings for intuitive search.
- **Visual Similarity Search**: Find related images efficiently from a reference image.
- **OCR-Assisted Retrieval**: Search through screenshots, documents, and images containing text.
- **Video Frame Extraction and Indexing**: Enable semantic search across video assets by analyzing extracted frames.
- **Multi-Source Media Support**: Utilize source-aware media identifiers for robust indexing.
- **Google Drive Integration**: Authenticate via OAuth to seamlessly index and serve cloud-based images.
- **Interactive Web Interface**: Browse results, preview media, and explore relationships in an intuitive graph-based view.
- **gRPC Inference Service**: Standalone CLIP embedding and OCR extraction server for polyglot services (Go, Rust, etc.).
- **Configurable Indexing Behavior**: Customize settings effortlessly through the desktop application.

## How It Works

### High-Level Workflow

<p align="center">
  <img src="UI/Semantixel WebUI/assets/architecture.png" alt="SemantiXel Architecture" width="600px" height="800px"/>
</p>

Semantixel integrates three distinct retrieval strategies:

- **Visual Retrieval**: Embeds image and text queries into a shared CLIP space.
- **OCR Retrieval**: Extracts and indexes text from images for semantic and BM25 search.
- **Metadata-Aware Serving**: Resolves indexed items via source-aware media identifiers instead of relying purely on local file paths.

At a high level:
1. Media is discovered from configured local directories and connected sources.
2. Images and extracted video frames are embedded utilizing CLIP.
3. OCR text is extracted and stored for both semantic and BM25 search mechanisms.
4. Embeddings and metadata are managed within ChromaDB and the BM25 index.
5. The REST API serves search results and media content directly to the web UI.
6. A gRPC inference server runs alongside Flask, exposing CLIP and OCR as language-agnostic RPCs for future Go services.

## Requirements

- Python 3.11
- CUDA-capable GPU (Recommended for optimal indexing and search performance)
- Conda or an alternative Python environment manager
- `grpcio` and `grpcio-tools` (included in `requirements.txt`)

## Installation and Setup

### Environment Setup

Create and activate a new environment:

```bash
conda create -n semantixel python=3.11 -y
conda activate semantixel
pip install -r requirements.txt
```

### Application Configuration

Launch the settings utility:

```bash
python settings.py
```

### Execution

Run a full local scan to index your files:

```bash
python main.py --scan
```

Start the application server:

```bash
python main.py --serve
```

Start the gRPC inference server (separate process — must run alongside Flask if Go services are consuming it):

```bash
python main.py --grpc
```

Alternatively, execute the default combined workflow:

```bash
python main.py
```

## Configuration Options

Runtime configuration is maintained in `config.yaml`. Key settings include:

- `include_directories`: Local directories to scan.
- `exclude_directories`: Local directories to ignore.
- `batch_size`: The number of items processed per indexing batch.
- `clip`: Configuration for the CLIP provider and model checkpoints.
- `text_embed`: Settings for the text embedding provider.
- `ocr_provider`: Selection of the OCR backend.
- `google_drive`: Configuration for Google Drive integration.

## Google Drive Integration

Semantixel natively supports indexing and serving images directly from Google Drive.

**Example Configuration:**

```yaml
google_drive:
  enabled: true
  client_secret_file: path/to/client_secret.json
  token_file: google_drive_token.json
  redirect_uri: http://localhost:23107/integrations/google_drive/auth/callback
  folder_ids: []
  include_shared_drives: false
  page_size: 100
```

**Integration Steps:**
1. Create a Google Cloud OAuth client of type `Web application`.
2. Set the redirect URI to `http://localhost:23107/integrations/google_drive/auth/callback`.
3. Download the client secret JSON file.
4. Update `config.yaml` accordingly.
5. Start the application and authenticate via `Connect Google Drive` in the web UI.
6. Run `python main.py --scan` to commence indexing Drive images.

*Note: OAuth secrets and token files must remain secure and excluded from version control.*

## Search Modalities

Semantixel supports comprehensive search capabilities:

- **Caption Search**: Retrieve images or video frames using natural language descriptions.
- **Similar Image Search**: Discover visually related images starting from a reference image or identifier.
- **Text Content Search**: Locate images based on OCR-detected text.
- **Graph Exploration**: Analyze and inspect similarity relationships between indexed assets.

## Sample Use Cases

- Query a screenshot archive using natural language such as "dashboard with a warning banner" or "terminal output showing build failure".
- Locate visually similar product photography, design mockups, or duplicate assets across a large catalog.
- Identify images containing specific OCR-detected phrases like invoice numbers, application labels, or error messages.
- Navigate significant segments within video files by retrieving semantically relevant extracted frames.
- Establish a unified knowledge base combining local storage with cloud-hosted libraries.

## Repository Structure

Key directories:

- `semantixel/`: Core backend services, API endpoints, providers, source integrations, and gRPC server.
- `settings/`: Desktop configuration interface.
- `UI/`: Web interface and Flow Launcher integrations.
- `proto/`: Protobuf contract for the gRPC inference service.
- `scripts/`: Utility scripts (e.g., proto stub generation).
- `docs/`: Technical documentation and system design notes.
- `db/`: ChromaDB and BM25 artifacts generated during runtime.

## Security Overview

- Local media access is strictly confined to configured inclusion directories.
- External URLs are rigorously validated prior to ingestion for image queries.
- Google Drive access is securely delegated via OAuth 2.0 API calls.
- Security credentials and tokens must remain local and ignored by Git.
