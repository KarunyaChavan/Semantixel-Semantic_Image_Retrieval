# Semantixel

Semantixel is a semantic media retrieval system for local and connected image sources. It indexes visual content with CLIP embeddings, extracts on-image text with OCR, and exposes search workflows for text-to-image, image-to-image, and OCR-backed retrieval through a lightweight web interface.

The project is designed for personal knowledge bases, research datasets, screenshot archives, and media collections where keyword search alone is not sufficient.

Get more familiar with Semantixel by knowing its purpose and what it offers:

<p align="center">
  <a href="UI/Semantixel WebUI/assets/SemantiXel__AI_Image_Search.mp4">
    <img src="UI/Semantixel WebUI/assets/icon.png" width="200" alt="Watch Demo Video"/>
  </a>
  <br/>
  <strong><a href="UI/Semantixel WebUI/assets/SemantiXel__AI_Image_Search.mp4">▶️ Watch Demo Video</a></strong>
</p>

## Features

- Natural-language image retrieval using CLIP text and image embeddings
- Visual similarity search for finding related images from a reference image
- OCR-assisted retrieval for screenshots, documents, and images containing text
- Video frame extraction and indexing for semantic search across video assets
- Multi-source media support through source-aware media identifiers
- Google Drive image integration with OAuth-based authorization and serving
- Web interface for result browsing, previewing, and graph-based exploration
- Configurable indexing behavior through the desktop settings application

## How It Works

### High-level workflow

<p align="center">
  <img src="../UI/Semantixel WebUI/assets/architecture.png" alt="SemantiXel Logo" width="600px" height="800px"/>
</p>

Semantixel combines three retrieval strategies:

- Visual retrieval: image and text queries are embedded in a shared CLIP space
- OCR retrieval: text extracted from images is indexed for semantic and keyword lookup
- Metadata-aware serving: indexed items are resolved through source-aware media identifiers instead of assuming a local file path

At a high level:

1. Media is discovered from configured local directories and optional connected sources.
2. Images and extracted video frames are embedded with CLIP.
3. OCR text is extracted and stored for semantic and BM25 search.
4. Embeddings and metadata are stored in ChromaDB and the BM25 index.
5. The Flask API serves search results and media content to the web UI.

## Requirements

- Python 3.11
- CUDA-capable GPU recommended for indexing and search performance
- Conda or another Python environment manager

Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Setup

Create and activate an environment:

```bash
conda create -n semantixel python=3.11 -y
conda activate semantixel
pip install -r requirements.txt
```

Configure the application:

```bash
python settings.py
```

Run a full local scan:

```bash
python main.py --scan
```

Start the server:

```bash
python main.py --serve
```

Or run the default combined flow:

```bash
python main.py
```

## Configuration

Runtime configuration is stored in `config.yaml`. Core settings include:

- `include_directories`: local directories to scan
- `exclude_directories`: local directories to skip
- `batch_size`: indexing batch size
- `clip`: CLIP provider and checkpoint settings
- `text_embed`: text embedding provider settings
- `ocr_provider`: OCR backend selection
- `google_drive`: optional Google Drive integration settings

## Google Drive Integration

Semantixel can index and serve images from Google Drive in addition to local files.

Example configuration:

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

Setup flow:

1. Create a Google Cloud OAuth client of type `Web application`
2. Configure the redirect URI as `http://localhost:23107/integrations/google_drive/auth/callback`
3. Download the client secret JSON file
4. Update `config.yaml`
5. Start the application and use `Connect Google Drive` in the web UI
6. Run `python main.py --scan` to index Drive images

Notes:

- The current connector targets Google Drive images first
- The integration uses OAuth and stores a local token file for subsequent access
- OAuth secrets and token files should not be committed to source control

## Search Modes

Semantixel currently supports:

- Caption search: retrieve images or video frames from natural language descriptions
- Similar image search: retrieve visually related images from a reference image or media identifier
- Text content search: retrieve images based on OCR-detected text
- Graph exploration: inspect similarity relationships between indexed items

## Sample Use Cases

- Search a screenshot archive with natural language queries such as "dashboard with a warning banner" or "terminal output showing build failure"
- Retrieve visually similar product photos, design references, or duplicate assets from a large image collection
- Find images that contain specific OCR-detected phrases such as invoice numbers, UI labels, or error messages
- Explore important moments inside video files by retrieving semantically relevant extracted frames
- Build a personal or team knowledge base that combines local folders with cloud-hosted image libraries
- Extend semantic search beyond local storage by connecting Google Drive as an additional media source

## Repository Structure

Key directories:

- `semantixel/`: backend services, API, providers, and source integrations
- `settings/`: desktop configuration UI
- `UI/`: web interface and Flow Launcher integration
- `docs/`: technical documentation and design notes
- `db/`: ChromaDB and BM25 artifacts created at runtime

## Security Notes

- Local media access is restricted to configured include directories
- External URL handling is validated before image-query ingestion
- Google Drive access is delegated through OAuth-backed API calls
- OAuth client secrets and token files should remain local and ignored by Git

