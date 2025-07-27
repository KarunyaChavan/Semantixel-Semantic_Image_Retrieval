# 👤 SemantiXel — Face Recognition Feature

The **Face Recognition** feature in **SemantiXel** extends semantic image search by enabling powerful facial search and recognition capabilities. Leveraging the [DeepFace](https://github.com/serengil/deepface) library with the **Facenet** model, this module allows users to identify, match, and explore faces across large image datasets — all seamlessly integrated with semantic queries.

---

## 🎯 Overview

This module enables face-based search using deep learning models, enhancing media understanding within the SemantiXel ecosystem. It allows:
- Identification of individuals in images.
- Cross-querying via both face and semantic contexts.
- Real-time interaction through a web-based interface.

---

## 🧰 Setup & Configuration

### 1. 🔧 Installation

Ensure `DeepFace` is included in your `requirements.txt`:
```bash
pip install deepface
````

Install all project dependencies:

```bash
pip install -r requirements.txt
```

### 2. ⚙️ Configuration

All paths and parameters are defined in `config.yaml`, including:

* Image directory paths
* CLIP model parameters
* OCR settings
* Face data storage locations

### 3. 📁 Directory Structure

| Directory                 | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| `face_data/`              | Stores processed face images and metadata    |
| `face_db/known_faces.pkl` | Serialized database of known face embeddings |

---

## 🛠️ Code Components

### 1. `face_encoder.py`

Encodes known face images into embeddings using the Facenet model. These embeddings are stored for later comparison.

```python
DeepFace.represent(img_path, model_name="Facenet")
```

### 2. `face_search.py`

Handles querying: compares uploaded/query face embeddings to known faces and returns matches with similarity scores.

### 3. `integrated_search.py`

Combines face search with CLIP-based semantic search, enabling multi-modal queries such as:

> *"Find images of John smiling at a beach."*

### 4. `server.py`

Flask-powered backend providing RESTful API endpoints for face search, integrated queries, and UI communication.

---

## 🌐 User Interface

The frontend is built with JavaScript and connects to Flask endpoints. Users can:

* Upload an image for face search.
* Input names or descriptions.
* Receive face + semantic search results in real time.

```html
<form id="search-form">
  <input type="file" id="face-image" />
  <input type="text" id="query" placeholder="Search faces or scenes..." />
</form>
```

---

## 🔍 Key Features

✅ **Face Encoding**
Transforms face images into 128-dimensional embeddings using Facenet.

✅ **Face Search**
Finds visually similar faces via vector comparison using cosine similarity.

✅ **UI Integration**
Smooth user experience through interactive queries and image previews.

✅ **Multi-modal Search**
Enables combined face + semantic retrieval for deeper media insights.

---

## 🚀 Getting Started

```bash
# Step 1: Encode faces
python face_encoder.py

# Step 2: Run the Flask server
python server.py
```

Once the server is running, visit `http://localhost:5000` to use the UI.

---

## 📦 Dependencies

* Python 3.9+
* DeepFace
* Flask
* CLIP
* NumPy, OpenCV, etc.

Make sure all dependencies are listed in `requirements.txt`.

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
