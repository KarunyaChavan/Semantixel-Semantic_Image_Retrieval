# gRPC Inference Service

This document explains the gRPC inference service — what it is, why it exists, and how to use it.

## What is it?

A standalone gRPC server that runs alongside the existing Flask app. It exposes CLIP (image/text embeddings) and OCR (text extraction) as network RPCs instead of keeping them locked inside the Flask process.

## Why gRPC?

| Problem | Solution |
|---|---|
| Flask is Python-only — Go services cannot call it directly | gRPC is language-agnostic; Go, Rust, etc. can generate stubs from the same `.proto` file |
| ML models (CLIP, OCR) are coupled to the web server | The inference service runs independently — restarting Flask won't reload models |
| JSON serialization is slow for float arrays (embeddings) | Protobuf binary format is ~10x faster for vector data |
| No health/readiness endpoint for ML models | The `HealthCheck` RPC reports model load status and device |

## Architecture

```
┌─────────────────┐     gRPC (port 50051)     ┌──────────────────────┐
│  Go Scanner     │ ────────────────────────── │  Python gRPC Server  │
│  (Issue #19)    │    EmbedImage, EmbedText,  │  (CLIP + OCR models) │
│  Go GraphQL GW  │    ExtractOCR, HealthCheck │                      │
│  (Issue #21)    │ ────────────────────────── │  shared ModelManager │
└─────────────────┘                            └──────────────────────┘
                                                         ▲
                                                         │ same singleton
                                                         ▼
┌─────────────────┐                            ┌──────────────────────┐
│  Flask Web UI   │ ◄────── REST ───────────── │  Existing Flask App  │
│  (unchanged)    │                            │  (uses same models)  │
└─────────────────┘                            └──────────────────────┘
```

Both Flask and the gRPC server use the same `ModelManager` singleton, so models are loaded at most once regardless of which layer accesses them.

## RPCs

| RPC | Input | Output | What it does |
|---|---|---|---|
| `EmbedImage` | One or more image bytes | L2-normalized embeddings + model name + dimension | CLIP image embedding |
| `EmbedText` | One or more text strings | L2-normalized embeddings + model name + dimension | CLIP text embedding (batched) |
| `ExtractOCR` | One or more image bytes + optional threshold (0.0–1.0) | `OCRResult` per image (extensible wrapper) | OCR text extraction |
| `HealthCheck` | Empty | `ServingStatus` enum + model info + device | Readiness probe |

## How to run

```bash
# Start the gRPC server (default port: 50051)
python main.py --grpc

# Or with a custom port
python main.py --grpc --grpc-port 50052

# Or directly
python -m semantixel.grpc_server
```

The Flask server continues to work as before:

```bash
python main.py --serve
```

Both can run simultaneously on different ports.

## Key proto design decisions

- **`ServingStatus` enum (not a free-form string):** `NOT_SERVING`, `LOADING`, `SERVING` are typed in the proto, eliminating string comparison bugs.
- **`model_name` and `embedding_dim` in responses:** Consumers know exactly which model produced the embedding and what dimensionality to expect without hardcoding.
- **`EmbedTextRequest` is batched (`repeated string texts`):** Symmetrical with `EmbedImageRequest`. Both image and text embedding accept multiple inputs for bulk indexing pipelines.
- **`optional float threshold`:** The field uses `optional` so the server can distinguish between "client omitted it" (applies 0.4) and "client explicitly set 0.0" (uses 0.0, meaning no filtering). Validation ensures the value is in `[0.0, 1.0]`.
- **`OCRResult` wrapper:** Instead of returning raw strings, OCR returns `repeated OCRResult` messages. This allows adding per-result fields (`confidence`, `token_count`, etc.) later without breaking the API shape.

## Proto contract

The service definition lives in `proto/semantixel_inference.proto`. This file is the single source of truth — any language can generate client/server stubs from it.

```protobuf
service SemantixelInference {
  rpc EmbedImage(EmbedImageRequest) returns (EmbedImageResponse);
  rpc EmbedText(EmbedTextRequest) returns (EmbedTextResponse);
  rpc ExtractOCR(ExtractOCRRequest) returns (ExtractOCRResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}
```

## Regenerating stubs

The generated Python stubs (`semantixel/*_pb2*.py`) are gitignored and must be regenerated after modifying the `.proto` file:

```bash
python scripts/generate_proto.py
```

## Consuming from Go (example)

```go
import pb "github.com/yourorg/semantixel/proto"

conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewSemantixelInferenceClient(conn)

// Embed an image
imgResp, _ := client.EmbedImage(ctx, &pb.EmbedImageRequest{
    Images: [][]byte{imageBytes},
})
embedding := imgResp.Embeddings[0].Values // []float32
log.Printf("model=%s dim=%d", imgResp.ModelName, imgResp.EmbeddingDim)

// Embed multiple texts (batched)
txtResp, _ := client.EmbedText(ctx, &pb.EmbedTextRequest{
    Texts: []string{"cat", "dog", "car"},
})
for i, emb := range txtResp.Embeddings {
    log.Printf("text[%d] dim=%d", i, len(emb.Values))
}

// Health check with enum
health, _ := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
if health.Status == pb.ServingStatus_SERVING {
    log.Println("server is ready")
}

// OCR with optional threshold
ocrResp, _ := client.ExtractOCR(ctx, &pb.ExtractOCRRequest{
    Images: [][]byte{imageBytes},
})
for _, r := range ocrResp.Results {
    fmt.Println(r.Text)
}
```

## Validation checklist

After making changes to the gRPC service:

- [ ] `python scripts/generate_proto.py` runs without errors
- [ ] Generated stubs are importable: `from semantixel import semantixel_inference_pb2`
- [ ] Server starts: `python main.py --grpc`
- [ ] `HealthCheck.status` returns the `ServingStatus` enum (not a string)
- [ ] `HealthCheck` returns `SERVING` once models are warm
- [ ] `EmbedImageResponse.model_name` and `embedding_dim` are populated
- [ ] `EmbedTextResponse.model_name` and `embedding_dim` are populated
- [ ] `EmbedTextRequest` accepts multiple texts (`repeated string`)
- [ ] `EmbedTextResponse.embeddings` preserves input order
- [ ] Embedding dimensions match the original Flask outputs
- [ ] `ExtractOCR` rejects threshold outside `[0.0, 1.0]`
- [ ] `ExtractOCR` uses `OCRResult` wrapper (not raw strings)
- [ ] `optional` threshold distinguishes unset from explicit 0.0
- [ ] OCR outputs match the original Flask outputs
