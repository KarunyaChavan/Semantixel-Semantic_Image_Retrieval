# Future Work and Improvements

This document outlines possible directions to extend and improve Semantixel.

## Short-term improvements

- Add FAISS/Milvus integration for large-scale ANN search, currently relying on Chroma.
- Improve UI: add pagination, result clustering, and richer metadata views.
- Add unit and integration tests for indexing and retrieval pipelines.

## Mid-term features

- Add a feature where user can remark on result, so that model can learn online.
- Multimodal query blending: support combined text + image queries.
- Fine-tune CLIP on domain-specific pairs to improve retrieval relevance.

## Long-term research ideas

- Multilingual text encoder support.
- Cross-modal re-ranking using a separate transformer-based ranker.
- Active learning loop to incorporate user feedback into improved embeddings.

💡 Note: Prioritize integration with a scalable vector index if dataset grows beyond a few hundred thousand images.
