# Data Pipeline and Preprocessing

This document explains how images are discovered, preprocessed, and prepared for embedding in Semantixel.

## Dataset ingestion

- Image discovery scripts live in `Index/` (e.g., `scan.py`, `scan_default.py`). They iterate directories and produce CSVs or lists of file paths (`Index/paths.csv`).
- Metadata (if available) is read alongside image files and stored in the index database.

## Image preprocessing

- Images are loaded and resized to the model's expected input size (for CLIP models, typically 224x224 or 336x336 for larger models).
- Standard normalization (mean/std) matching the pretrained model is applied.
- Optional augmentations (for training only): random crop, flip, color jitter, and normalization.

## Embedding caching and storage

- Embeddings are computed in batches and cached to disk or the `db/` directory. The repository contains `db/chroma.sqlite3` and per-index folders.
- `create_index.py` orchestrates the indexing: it loads images, computes embeddings via `CLIP/` or `text_embeddings/`, and writes vectors and metadata to the database.

⚙️ Example:

1. Scan images: `python Index/scan.py --root <image_folder>`
2. Create index: `python create_index.py --data Index/paths.csv --out db/`

💡 Note: Batching (e.g., 32–128 images per batch) reduces memory overhead and speeds up GPU utilization.
