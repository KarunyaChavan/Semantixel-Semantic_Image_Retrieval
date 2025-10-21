# System Architecture

This document outlines the high-level architecture of Semantixel and how the main components interact.

## High-level workflow

<p align="center">
  <img src="UI/Semantixel WebUI/assets/architecture.png" alt="SemantiXel Logo" width="600px" height="800px"/>
</p>

1. Data ingestion: images and optional metadata are discovered and scanned by the `Index/` utilities (see `scan.py`, `scan_default.py`).
2. Feature extraction: images are preprocessed and passed to the embedding model in `CLIP/` or `text_embeddings/` modules to produce fixed-length vectors.
3. Index creation: embeddings are stored in an index (on-disk DB or vector index). The repository includes `create_index.py` and `Index/create_db.py` to build and persist the index.
4. Search API / UI: `server.py` or the WebUI under `UI/Semantixel WebUI/` provide endpoints and pages for text/image queries and result visualization.

## Component interactions

- Indexing tools (Index/*, `create_index.py`) read images and metadata, call the embedding pipeline, and write vectors to persistent storage in `db/`.
- Embedding modules (`CLIP/`, `text_embeddings/`) expose encoders to convert images or text to vectors.
- The retrieval layer (`server.py`, `main.py`, `face_search.py`) loads the vector store and performs nearest-neighbor lookup and filtering.
- UI (`UI/Semantixel WebUI/`, Flow Launcher plugin) communicates with the backend endpoints to present ranked images and allow interactive refinement.

## Libraries and frameworks

- PyTorch (for model inference and optionally fine-tuning).
- Hugging Face transformers / custom wrapper (`CLIP/hftransformers_clip.py`) for model loading.
- Chroma/SQLite (the repository contains `db/chroma.sqlite3`) or other vector DBs for persistent storage.
- Web UI: simple static HTML/CSS/JS under `UI/Semantixel WebUI/` and optional Flow Launcher integration.
- Optional: FAISS or other vector search libraries can be integrated for scalable ANN search.

⚙️ Example block diagram (logical):

Image files -> Index scanner -> Preprocessing -> CLIP encoder -> Embeddings -> Index persisted (db/) -> Search API -> UI
