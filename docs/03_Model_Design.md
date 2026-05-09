# Model Design

This project utilizes CLIP-style models (Contrastive Language-Image Pretraining) as the primary embedding mechanism. CLIP establishes a joint embedding space where semantically similar images and text are situated closely together.

## Embedding model architecture

- Image encoder: Typically a Vision Transformer (ViT) or convolutional backbone that generates a robust feature vector.
- Text encoder: A Transformer-based text encoder that produces a textual embedding of the same dimensionality.
- Projection heads: Both image and text features are linearly projected into a shared space and normalized.

The repository includes modular providers in `semantixel/providers/`:

- `clip/hf_provider.py`: Utilities to wrap Hugging Face CLIP checkpoints for seamless inference.
- `text/hf_provider.py`: Handles standalone text embeddings.
- `ocr/doctr_provider.py`: Manages Optical Character Recognition to extract text from images.

## Feature extraction and similarity

- Extracted features are normalized using L2-normalization before similarity computations.
- The primary similarity metric is cosine similarity, which is effectively calculated via the dot product of the normalized vectors to rank results efficiently.

## Extensibility and Fine-Tuning

While the core modules are focused on highly optimized inference and scalable indexing, the provider architecture allows easy integration of new models or fine-tuned checkpoints. Standard strategies for performance enhancement include:

- Contrastive fine-tuning on domain-specific datasets.
- Utilizing data augmentation strategies for domain adaptation.

Note: The modular structure inside `semantixel/providers/` ensures that switching model backends requires minimal changes to the core search and indexing services.
