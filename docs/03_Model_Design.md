# Model Design

This project uses CLIP-style models (Contrastive Language–Image Pretraining) as the core embedding mechanism. CLIP learns a joint embedding space where images and text with similar semantic content are close together.

## Embedding model architecture

- Image encoder: typically a convolutional or Vision Transformer backbone (ResNet, ViT) that outputs a feature vector.
- Text encoder: a Transformer-based text encoder that outputs a textual embedding of the same dimensionality.
- Projection heads: both image and text features are linearly projected into a shared embedding space and normalized.

The repository includes wrappers in `CLIP/`:

- `hftransformers_clip.py` — utilities to wrap Hugging Face CLIP model checkpoints for inference.
- `mobile_clip.py` — a lightweight/mobile friendly interface.

## Feature extraction and similarity

- Features are normalized (L2-normalization) before similarity computation.
- Similarity metric: cosine similarity between normalized vectors is used for ranking, implemented as a dot product after normalization.

## Training and fine-tuning

This repository is primarily focused on inference and indexing. If fine-tuning is required, recommended strategies include:

- Contrastive fine-tuning on domain-specific image-text pairs.
- Linear-probing the projection head with a small learning rate and warm restarts.
- Data augmentation for images (random crops, color jitter) to improve invariance.

💡 Note: Training requires labelled image–text pairs and GPU resources. The codebase contains inference wrappers; see `text_embeddings/` and `CLIP/` for model loading and usage.
