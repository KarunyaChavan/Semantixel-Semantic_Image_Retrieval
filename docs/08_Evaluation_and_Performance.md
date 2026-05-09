# Evaluation and Performance

This document details evaluation methodologies and performance targets for Semantixel deployments.

## Evaluation metrics

- Precision@K: The proportion of highly relevant images present within the top-K returned results.
- Recall@K: The fraction of all relevant items successfully retrieved within the top-K.
- Mean Reciprocal Rank (MRR): Useful for evaluating the rank of the first relevant result.

## Latency and throughput

- Target end-to-end latency: For interactive web sessions, query processing and retrieval should complete in under 300ms on a moderately sized index.
- Throughput optimization: Batching is strictly implemented during the indexing phase to ensure high throughput on hardware accelerators.

## Scalability

- The default architecture utilizing ChromaDB is highly optimized for local to medium-scale deployments (up to several million vectors).
- Lexical search scalability is maintained through the efficient `bm25_service.py`.

## Hardware notes

- GPU Acceleration: Utilizing a CUDA-enabled GPU is strongly recommended. It drastically reduces the inference time for both CLIP embeddings and DocTR OCR processing.
- CPU-only Execution: Supported for environments without dedicated accelerators, though it is primarily suited for smaller datasets or background processing where latency is not critical.

Performance is largely dictated by the selected model backends in the providers directory and the available compute resources.
