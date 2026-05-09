# Glossary

This glossary defines technical terminology and concepts utilized throughout the Semantixel documentation and codebase.

- CLIP: Contrastive Language-Image Pretraining. A model architecture that aligns images and text into a shared latent space, critical for zero-shot semantic retrieval.
- Embedding: A high-dimensional, fixed-length numerical vector representing the semantic meaning of an image or text sequence.
- Cosine Similarity: A metric utilized to determine the similarity between two vectors, effectively calculated via the dot product when vectors are normalized.
- L2-Normalization: The process of scaling a vector such that its Euclidean norm is exactly 1.
- Vector Database: A specialized data storage system optimized for the rapid retrieval of high-dimensional embeddings (e.g., ChromaDB).
- BM25: Best Matching 25. A probabilistic ranking function used for lexical search over text data, utilized by `bm25_service.py` for OCR retrieval.
- OCR: Optical Character Recognition. The process of extracting readable text from image data, handled by `doctr_provider.py`.
- Precision@K: An evaluation metric indicating the fraction of retrieved items within the top-K results that are genuinely relevant.
- Recall@K: A metric indicating the fraction of all relevant items successfully retrieved within the top-K results.
- Top-K: The specific number of highest-scoring results returned by the search service.
- WSGI: Web Server Gateway Interface. The standard Python interface utilized in `wsgi.py` to serve the API.

Note: For concrete implementation details, inspect core services such as `semantixel/services/search_service.py` and the provider abstractions in `semantixel/providers/`.
