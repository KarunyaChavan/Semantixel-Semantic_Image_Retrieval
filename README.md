# 📸 SemantiXel — Semantic Image Retrieval

---

SemantiXel is a lightweight and modern web-based interface for performing **semantic search on image datasets** using CLIP and sentence embeddings. It enables intelligent retrieval of images based on **text queries**, **image similarity**, or **embedded textual content**, all in an elegant UI built for clarity and speed.

> ✨ Designed for creators, researchers, and developers to explore semantic media understanding with ease.

---

## 📚 Documentation

For a detailed technical overview, architecture, setup instructions, and advanced usage, see the [`docs/`](docs/) directory. It contains:

- System architecture and workflow
- Model and embedding details
- Data pipeline, search logic, UI/API, deployment, and more
- A glossary of key terms

Refer to these docs for in-depth understanding and implementation guidance.
### Quick Setup
1. Create a virtual environment:
   ```
   conda create -n Semantixel python=3.11 -y
   conda activate semantixel
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure Settings:
   ```
   python settings.py
   ```

4. Run: (Creates Index + Runs Server + Launches UI)
   ```
   python main.py
   ```
   
---

## 🏛️ Architecture

<p align="center">
  <img src="UI/Semantixel WebUI/assets/architecture.png" alt="SemantiXel Logo" width="600px" height="800px"/>
</p>

---

---

## 🚀 Features

- 🔍 **Text-to-Image Search** using CLIP (`openai/clip-vit-base-patch32`)
- 🖼️ **Image-to-Image Similarity Search** via vision embeddings
- 📝 **Embedded Text Search** for documents and OCR content
- 🎛️ Customizable `threshold` and `top-K` ranking
- 💻 Fast, responsive UI with a clean white theme
- 🧠 Powered by HuggingFace Transformers & Doctr OCR
- 📂 Supports directory-level image indexing

---

## 🖼️ Sample Use Cases

- Retrieve screenshots showing "Apple Intelligence" in YouTube thumbnails
- Find similar photos or memes from your collection
- Detect specific phrases or embedded text in image-based documents
- Build your own AI-powered personal visual library

---


