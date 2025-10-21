# Semantic Search and Retrieval

This document explains how Semantixel performs semantic matching and returns ranked images for a given query.

## Query types

- Semantic Queries: Intent/meaning based search, mostly similar to image captioning.
- Text queries: convert text to an embedding using the text encoder and perform nearest-neighbor search.
- Image queries: compute the image embedding and search for nearest images in the index.

## Similarity metric and ranking

- Metric: cosine similarity between normalized embeddings (implemented as dot product).
- Ranking: results are sorted by descending similarity score. The system returns top-K results (configurable, e.g., `top_k` parameter in UI).

## Filtering and thresholds

- The UI exposes options to filter results (e.g., minimum similarity threshold, face-based filtering) if metadata is available.
- A threshold slider (in the WebUI) can be used to exclude low-similarity results.

## Performance considerations

- For small/medium datasets (<100k vectors) an SQLite-based approach or Chroma may suffice.
- For >100k vectors, use an ANN library (FAISS, hnswlib) with appropriate indexing (IVF, HNSW) for fast queries.

⚙️ Example ranking logic (pseudocode):

```py
q = text_encoder.encode(query_text)
q = q / q.norm()
scores = index.dot(q)  # vectorized dot product against normalized database
top_idx = scores.argsort(descending=True)[:top_k]
```
