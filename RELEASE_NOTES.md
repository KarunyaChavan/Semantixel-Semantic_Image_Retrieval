# 📸 SemantiXel v1.0.0 — Release Notes

**Release Date:** October 2025

---

## 🎉 Overview

SemantiXel 1.0.0 is the initial release of a powerful semantic image retrieval system that brings intelligent visual search capabilities to your fingertips. Built with state-of-the-art AI models and a modern web interface, SemantiXel enables you to search through your image collections using natural language, visual similarity, embedded text, and even facial recognition.

---

## ✨ Key Features

### 🔍 Semantic Search Capabilities

- **Text-to-Image Search**
  - Search for images using natural language descriptions
  - Powered by OpenAI CLIP (`openai/clip-vit-base-patch32`)
  - Understands context and semantic meaning beyond keywords
  - Example: "sunset over mountains" or "people playing basketball"

- **Image-to-Image Similarity Search**
  - Find visually similar images from your collection
  - Uses CLIP vision embeddings for deep visual understanding
  - Perfect for finding duplicate images or similar compositions

- **Embedded Text Search (OCR)**
  - Search for text content within images
  - Powered by Doctr OCR engine
  - Ideal for screenshots, documents, and image-based text
  - Uses sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`)

- **Face Recognition & Search**
  - Identify and search for specific people in your images
  - Integrated face-semantic search: "Find [person] playing cricket"
  - Combines facial recognition with activity/context understanding
  - Powered by DeepFace and MTCNN

### 🖥️ User Interfaces

- **Modern Web UI**
  - Clean, responsive design with white theme
  - Real-time search with adjustable parameters
  - Threshold and top-K ranking controls
  - Support for local images and URL-based searches
  - Visual results display with similarity scores

- **Flow Launcher Plugin**
  - Desktop integration for Windows users
  - Quick access to semantic search from your launcher
  - Seamless workflow integration

### 🧠 Advanced Configuration

- **Multiple AI Provider Support**
  - **CLIP Providers:**
    - HuggingFace Transformers CLIP
    - MobileClip (optimized for mobile/edge devices)
  - **Text Embedding Providers:**
    - HuggingFace Transformers
    - LlamaCPP (local GGUF models)
    - Ollama (local LLM integration)
    - OpenAI API (optional)

- **Flexible Indexing**
  - Directory-based image indexing
  - Deep scan mode for recursive directory search
  - Include/exclude directory patterns
  - Configurable batch processing
  - Automatic index cleaning and maintenance

- **Performance Options**
  - Batch processing support (configurable batch size)
  - ChromaDB vector database for fast retrieval
  - Persistent index storage
  - Optimized embedding generation

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Sufficient disk space for model weights (~1-2GB)
- Supported OS: Windows, macOS, Linux

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KarunyaChavan/Semantixel-Semantic_Image_Retrieval.git
   cd Semantixel-Semantic_Image_Retrieval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your image directories:**
   Edit `config.yaml` to specify your image directories:
   ```yaml
   include_directories:
     - /path/to/your/images
   ```

4. **Create the search index:**
   ```bash
   python create_index.py
   ```

5. **Start the web server:**
   ```bash
   python server.py
   ```

6. **Access the UI:**
   Open your browser and navigate to `http://localhost:5000`

### Quick Start Commands

- **Open settings:** `python main.py --settings`
- **Delete index:** `python main.py --delete-index`
- **Get index path:** `python main.py --get-index`
- **Open config file:** `python main.py --open-config-file`
- **Encode known faces:** `python main.py --encode-faces`

---

## 📦 Core Components

### Backend Architecture

- **Flask Server** (`server.py`)
  - RESTful API endpoints for all search operations
  - CORS support for web UI
  - Image parsing for local and remote images

- **Vector Database** (ChromaDB)
  - Efficient similarity search using HNSW algorithm
  - Separate collections for images and text embeddings
  - Persistent storage in `db/` directory

- **Indexing System** (`create_index.py`, `Index/`)
  - Multi-threaded image scanning
  - Automatic embedding generation
  - Index cleaning and validation

### Search Modules

- **CLIP Module** (`CLIP/`)
  - HuggingFace Transformers implementation
  - MobileClip support for edge deployment
  
- **Text Embeddings** (`text_embeddings/`)
  - Sentence transformer models
  - LlamaCPP integration
  - Ollama support

- **OCR Module** (`ocr_model/`)
  - Doctr OCR engine
  - Platform-specific optimizations (macOS support)

- **Face Recognition** (`face_recognition/`)
  - Face encoding and storage
  - Name-based face search
  - Integrated semantic + face search

---

## 🔧 Configuration

### config.yaml Options

```yaml
# Batch size for processing
batch_size: 8

# CLIP configuration
clip:
  HF_transformers_clip: openai/clip-vit-base-patch32
  mobileclip_checkpoint: mobileclip_s0
  provider: HF_transformers  # or mobileclip

# Text embeddings configuration
text_embed:
  HF_transformers_embeddings: sentence-transformers/all-MiniLM-L6-v2
  provider: HF_transformers  # or ollama, llamacpp, openai

# Scanning options
deep_scan: true  # Recursive directory search
scan_method: default

# OCR provider
ocr_provider: doctr

# Directories
include_directories:
  - /path/to/images
exclude_directories: []
```

---

## 📊 Technical Specifications

### Dependencies

- **Core:**
  - PyTorch >= 1.13.1
  - TorchVision >= 0.14.1
  - Transformers (HuggingFace)
  - ChromaDB 0.5.0

- **AI Models:**
  - open-clip-torch >= 2.20.0
  - python-doctr >= 0.8.1
  - DeepFace
  - MTCNN-OpenCV

- **Web & UI:**
  - Flask
  - Flask-CORS
  - ttkbootstrap
  - darkdetect

- **Utilities:**
  - Pillow
  - NumPy <= 1.26.4
  - PyYAML 6.0.1
  - requests
  - tqdm

### System Requirements

- **Memory:** Minimum 4GB RAM (8GB recommended)
- **Storage:** ~2GB for models, plus space for your image collection
- **GPU:** Optional but recommended for faster indexing (CUDA support)

---

## 🎯 Use Cases

1. **Personal Photo Library Management**
   - Search your vacation photos: "beach sunset"
   - Find similar compositions or duplicates
   - Locate photos of specific people

2. **Content Creation & Research**
   - Find reference images for design projects
   - Locate screenshots with specific content
   - Search meme collections by description

3. **Document Management**
   - Search scanned documents by text content
   - Find receipts, invoices, or notes by keywords
   - OCR-based text search in image archives

4. **Professional Workflows**
   - Product image search for e-commerce
   - Visual asset management for marketing teams
   - Research image database navigation

---

## 🛠️ API Endpoints

The Flask server exposes the following endpoints:

- `POST /search/text` - Text-to-image search
- `POST /search/image` - Image-to-image similarity search
- `POST /search/embedded-text` - OCR-based text search
- `POST /search/face` - Face recognition search
- `POST /search/integrated` - Combined face + semantic search
- `GET /images/<path>` - Serve indexed images

---

## 🐛 Known Issues & Limitations

- Face encoding requires manual setup of known faces in `face_data/`
- Large image collections (>10,000 images) may take significant time to index
- GPU acceleration recommended for collections over 5,000 images
- Some OCR models may not be optimized for all languages

---

## 🔮 Future Roadmap

- Multi-modal search combining multiple query types
- Additional CLIP model options (CLIP-B/16, CLIP-L/14)
- Video frame search capabilities
- Cloud deployment options
- Mobile app interface
- Incremental indexing for new images
- Advanced filtering options (date, location, file type)

---

## 👥 Contributing

We welcome contributions! Please see our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

---

## 📄 License

This project is licensed under the terms specified in [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

SemantiXel is built on top of amazing open-source projects:

- OpenAI CLIP
- HuggingFace Transformers
- ChromaDB
- Doctr OCR
- DeepFace
- And many more listed in requirements.txt

---

## 📞 Support

For issues, questions, or feature requests, please visit:
- GitHub Issues: [https://github.com/KarunyaChavan/Semantixel-Semantic_Image_Retrieval/issues](https://github.com/KarunyaChavan/Semantixel-Semantic_Image_Retrieval/issues)

---

**Happy Searching! 🎉**

*SemantiXel - Where pixels meet semantics*
