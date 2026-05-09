# Feature Extraction and Embedding

This document details the generation and management of feature embeddings.

## Embedding generation

- The system relies on providers located in `semantixel/providers/` to convert media and text into dense vectors.
- Typical lifecycle:
  - The service layer passes standardized media objects to the provider.
  - The image or text undergoes model-specific preprocessing (e.g., center crop, normalization, tokenization).
  - The model executes a forward pass, and the output vectors are projected and L2-normalized.

## Vector normalization and metrics

- Embeddings must be L2-normalized prior to storage. This guarantees that cosine similarity can be computed rapidly using a simple dot product operation.
- Search and ranking algorithms strictly depend on this normalized state.

## Storage and retrieval strategy

- Embeddings are persisted within a ChromaDB collection in the `db/` directory.
- The `index_service.py` manages the insertion of vector data, ensuring that standard identifiers and metadata payloads are bundled with the embeddings.
- Lexical data from OCR is concurrently stored in a BM25 index managed by `bm25_service.py`.

Note: Maintaining accurate metadata schemas alongside the vectors is critical for proper result filtering, sorting, and UI presentation during search operations.
