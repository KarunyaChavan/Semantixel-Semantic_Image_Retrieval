# System Architecture

This document outlines the high-level architecture of Semantixel and how the main components interact.

## High-level workflow

<p align="center">
  <img src="../UI/Semantixel WebUI/assets/architecture.png" alt="SemantiXel Architecture" width="600px" height="800px"/>
</p>

1. Data ingestion: Media files and optional metadata are discovered and processed by utilities in `semantixel/utils/` (such as `scan_utils.py` and `video_utils.py`) and integrated sources like Google Drive via `semantixel/sources/`.
2. Feature extraction: Images are preprocessed and passed to the embedding providers located in `semantixel/providers/` (e.g., `clip/hf_provider.py`, `ocr/doctr_provider.py`) to produce fixed-length vectors and extract text.
3. Index creation: Embeddings and text data are stored in a persistent index. `semantixel/services/index_service.py` handles the orchestration of writing vector data to ChromaDB and text data to a BM25 index.
4. Search API / UI: The API routes located in `semantixel/api/routes.py` (served via `wsgi.py`) expose endpoints for text/image queries. The WebUI under `UI/Semantixel WebUI/` visualizes the search results.

## Component interactions

- Scanning utilities read files and pass them to indexing services.
- The `index_service.py` coordinates with providers to extract features and persists the embeddings in the `db/` directory.
- Model providers (`semantixel/providers/`) expose clean interfaces to external ML models (Hugging Face, docTR, etc.).
- The retrieval layer (`semantixel/services/search_service.py`) queries the vector stores and BM25 index to perform nearest-neighbor lookup and hybrid search.
- The UI communicates with the backend API to present ranked results and enable interactive exploration.

## Libraries and frameworks

- PyTorch (for model inference).
- Hugging Face Transformers and Accelerate (managed via `semantixel/providers/clip/hf_provider.py`).
- ChromaDB for vector storage.
- Rank-BM25 for lexical search.
- Flask for serving the API layer.
- Web UI constructed with standard web technologies.

Example abstract workflow:
Image files -> Scanner -> Preprocessing -> Providers (CLIP/OCR) -> Services (Index/BM25) -> Persistent Storage (db/) -> API Routes -> UI
