# Data Pipeline and Preprocessing

This document outlines the pipeline for image discovery, preprocessing, and database ingestion within Semantixel.

## Dataset ingestion

- Discovery scripts are maintained in `semantixel/utils/` (e.g., `scan_utils.py` and `video_utils.py`). They recursively traverse local directories or connect to external sources like Google Drive via `semantixel/sources/google_drive_source.py`.
- Detected files are queued for processing, and relevant metadata (such as source URI, timestamps, and file sizes) is collated for indexing.

## Image preprocessing

- During processing, the `semantixel.media` module manages the loading and standardization of media assets.
- Images are resized and normalized according to the exact specifications expected by the active CLIP model provider.
- For video files, frame extraction routines capture keyframes for semantic analysis.

## Embedding caching and storage

- Embeddings are generated efficiently in batches via the `index_service.py` in `semantixel/services/`.
- Computed embeddings, OCR text, and structural metadata are stored persistently in the `db/` directory, managed by ChromaDB for vectors and a local BM25 index for text.

Example workflow:

1. Perform scan: The CLI initiates `scan_utils.py` to identify all relevant media.
2. Index generation: `index_service.py` coordinates with providers to encode the media and writes the results to the database.

Note: Batching is strictly enforced during the feature extraction phase to optimize GPU memory utilization and maximize throughput.
