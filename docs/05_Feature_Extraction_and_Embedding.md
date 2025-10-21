# Feature Extraction and Embedding

This document describes how features (embeddings) are produced and managed.

## Embedding generation

- Use the CLIP wrappers in `CLIP/` or the embedding modules under `text_embeddings/` to convert images (and text) to vectors.
- Typical workflow:
  - Preprocess image (resize, center crop, normalize).
  - Forward pass through image encoder.
  - Project and L2-normalize the output vector.

## Vector normalization and metrics

- Embeddings are L2-normalized before storage. This allows efficient cosine similarity computation using dot products.
- Similarity is computed as the dot product of normalized vectors (equivalent to cosine similarity).

## Storage and retrieval strategy

- The repo currently persists embeddings in `db/` (SQLite/Chroma shards). The `Index/create_db.py` script shows how metadata and vectors are written.
- For large-scale deployments, replace the storage layer with FAISS, Milvus, or an external vector DB and periodically sync the index.

⚙️ Example: normalize and store

```py
vec = model.encode_image(image_tensor)  # raw vector
vec = vec / vec.norm(dim=-1, keepdim=True)
# write vec to database with metadata
```

💡 Note: Keep the original image path and any textual metadata alongside the vector to enable result presentation and filtering.
