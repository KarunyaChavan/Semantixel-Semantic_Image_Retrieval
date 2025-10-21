# Glossary

This glossary defines technical terms and abbreviations used across the Semantixel documentation and codebase.

- CLIP: Contrastive Language–Image Pretraining — a model that learns a joint embedding space for images and text. Useful for zero-shot and semantic retrieval tasks.
- Embedding: a fixed-length numerical vector that represents semantic content of an image or text.
- Cosine similarity: a metric that measures the cosine of the angle between two vectors; commonly used for similarity in normalized embedding spaces.
- L2-normalization: scaling a vector so its Euclidean (L2) norm equals 1. Normalization enables cosine similarity to be computed via dot product.
- FAISS: Facebook AI Similarity Search — a library for efficient similarity search of dense vectors (supports exact and approximate search methods).
- ANN: Approximate Nearest Neighbors — algorithms (HNSW, IVF, PQ) that trade a small accuracy loss for large speedups on high-dimensional nearest neighbor search.
- Index: persistent store of vectors and associated metadata used for retrieval (can be a SQLite/Chroma store, FAISS index, or other vector DB).
- Chroma: an open-source vector database (and client libraries) used to store and query embeddings; the repo contains a `db/` folder with Chroma-like storage.
- Precision@K: the fraction of retrieved items in the top-K results that are relevant to the query.
- Recall@K: the fraction of all relevant items that are found in the top-K results.
- Top-K: the number of highest-ranked results returned for a query.
- Vector DB: specialized datastore optimized for storing and searching vector embeddings (FAISS, Milvus, Pinecone, Chroma, etc.).

💡 Note: If you see a term in the docs or comments that's unclear, search the codebase for its usage (e.g., `create_index.py`, `Index/create_db.py`, or `CLIP/`) to see concrete examples.
