# User Interface and API

This document describes the interaction layers for Semantixel, encompassing the REST API and the web frontend.

## Web UI

- Location: The frontend is hosted within `UI/Semantixel WebUI/`, utilizing a modern HTML, CSS, and JavaScript stack.
- The interface supports:
  - Natural language search input.
  - Direct image uploads for visual similarity queries.
  - Interactive controls for adjusting similarity thresholds and the number of results.
  - A responsive result grid with detailed metadata previews and an exploration graph.

Example client interaction:
1. The user inputs a query text or uploads a reference image.
2. The UI sends an asynchronous request to the backend API.
3. The API processes the query, searches the index, and returns structured JSON containing results and scores.
4. The client renders the results, updating the grid and exploration views.

## Backend / API

- The API is built with Flask and defined in `semantixel/api/routes.py`. It is served via `wsgi.py`.
- Key endpoints include:
  - `/api/search`: Accepts complex query payloads (text, image, threshold constraints) and returns ranked results.
  - `/api/index/status`: Provides diagnostic information regarding the database and indexed documents.
  - `/integrations/google_drive/*`: Handles OAuth flows and Drive-specific indexing commands.
- The API is designed to be fully decoupled from the underlying logic, relying strictly on the `search_service.py` and `index_service.py`.

Note: For a comprehensive understanding of request schemas and response formats, refer to the route definitions in `semantixel/api/routes.py`.
