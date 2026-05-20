# Semantic Search and Retrieval

This document explains the search modalities and ranking mechanics inside Semantixel.

## Query types

- Semantic Queries: Intent-based search leveraging natural language, functioning similarly to reverse image captioning.
- Lexical Queries: Text-based matching leveraging the BM25 index over extracted OCR data.
- Image Queries: Visual similarity search where an input image is embedded, and the nearest neighbors are retrieved from the index.

## Similarity metric and ranking

- The `search_service.py` orchestrates the retrieval process.
- The primary metric for vector retrieval is cosine similarity.
- Results from different modalities (e.g., CLIP and BM25) can be combined or filtered, and the system consistently returns a ranked list of the top-K results.

## Filtering and thresholds

- The search API supports extensive filtering based on metadata attributes (e.g., source type, exact matches).
- A configurable similarity threshold is enforced to prune low-confidence results before they are returned to the client.

## Performance considerations

- The backend leverages ChromaDB's optimized vector operations for responsive queries.
- `search_service.py` manages concurrent requests and handles embedding operations asynchronously where appropriate.

Example ranking logic execution:
The query is routed to the appropriate provider (e.g., `hf_provider.py`), encoded, and L2-normalized. The normalized vector is sent to the vector database, which computes the dot products and returns the highest scoring records.
