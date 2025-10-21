# User Interface and API

This document describes how users interact with Semantixel via the Web UI and any backend endpoints.

## Web UI

- Location: `UI/Semantixel WebUI/` contains a simple HTML/CSS/JS front-end (`index.html`, `styles.css`).
- The UI supports:
  - Text search input (natural language queries).
  - Image upload / image selection for visual queries.
  - Slider controls for similarity threshold and `top-k` results.
  - Result grid with clickable images (enlarge, view metadata).

⚙️ Example client flow:

1. User enters text or uploads an image.
2. Client sends the query to the backend (e.g., `server.py` REST endpoint or a local handler in `main.py`).
3. Backend computes embedding, queries index, and returns `top_k` image paths and scores.
4. Client renders images and allows user to select or refine the query.

## Backend / API

- `server.py` and `main.py` contain the entry points for running the server or offline scripts. Typical endpoints:
  - `/search` — POST endpoint that accepts `{ "text": "..." }` or `{ "image": <blob> }` and returns a JSON list of results.
  - `/health` — simple status check.

💡 Note: The exact endpoint names and parameters may vary; inspect `server.py` for the concrete implementation.
