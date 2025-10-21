# Evaluation and Performance

This document describes recommended evaluation metrics and performance considerations for Semantixel.

## Evaluation metrics

- Precision@K: proportion of relevant images in the top-K results.
- Recall@K: fraction of relevant items retrieved within top-K.
- Mean reciprocal rank (MRR) and average precision (AP) for ranked retrieval.

## Latency and throughput

- Measure end-to-end latency from receiving a query to returning top-K results. For interactive UIs target <300ms for small indices.
- Throughput: number of queries per second the system can sustain; GPU inference increases throughput for batch queries.

## Scalability

- Small datasets: SQLite/Chroma is acceptable.
- Large datasets: use FAISS or HNSW for approximate nearest neighbor (ANN) indexing. Use sharding and replication for distributed loads.

## Hardware notes

- GPU: reduces embedding inference time significantly — GPUs are recommended when indexing at scale or when providing low-latency interactive image queries.
- CPU-only: suitable for small datasets and offline indexing but will be significantly slower for large-scale indexing or realtime services.

⚙️ Example benchmark (sample):

| Setup | Embedding throughput | Search latency |
|---|---:|---:|
| RTX 3050 (batch=64) | 100 images/s | <15ms (FAISS) |
| CPU (8 cores) | ~100 images/s | ~70ms (small index) |
