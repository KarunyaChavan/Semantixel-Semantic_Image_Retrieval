"""REST API route definitions for the Semantixel server.

All search, indexing, and Google Drive OAuth endpoints are defined here
as a Flask :class:`Blueprint`.
"""

import io
import os
from flask import Blueprint, request, jsonify, send_file, send_from_directory, current_app, abort
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.core.security import is_safe_path, is_safe_url
from semantixel.media import describe_local_media, is_media_id, parse_media_id

main_bp = Blueprint("main", __name__)


def _validate_local_query_path(query: str) -> str:
    """Validate and return the locator for a local query path.

    Raises ``403`` if the path is outside the configured directories.
    """
    query_media = describe_local_media(query)
    if not is_safe_path(query_media.locator, config.include_directories):
        logger.warning("Path traversal attempt blocked: %s", query)
        abort(403, "Access to this path is forbidden.")
    return query_media.locator


# Search endpoints


@main_bp.route("/clip_text", methods=["POST"])
def clip_text():
    """Natural-language search across all indexed modalities.

    Request JSON:
        ``query`` (str): The search phrase.
        ``threshold`` (float, default 0): Minimum similarity score.
        ``top_k`` (int, default 5): Maximum results.
        ``media_type`` (str, default "image"): ``"image"``, ``"video"``,
            ``"audio"``, or ``"all"``.

    Returns:
        JSON array of results with metadata.
    """
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "image")
    try:
        results = current_app.search_service.semantic_text_search(query, top_k, threshold, media_type)
        return jsonify(results)
    except Exception as exc:
        logger.error("Semantic text search failed: %s", exc)
        return jsonify([])


@main_bp.route("/clip_image", methods=["POST"])
def clip_image():
    """Find visually similar images using CLIP.

    Request JSON:
        ``query`` (str): Local path, media ID, or HTTP URL to an image.
        ``threshold`` (float, default 0): Minimum similarity.
        ``top_k`` (int, default 5): Maximum results.
        ``media_type`` (str, default "all"): Type filter.

    Returns:
        JSON array of similar results, excluding the query itself.
    """
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "all")

    if query.startswith(("http://", "https://")):
        if not is_safe_url(query):
            abort(400, "Insecure URL provided.")
    else:
        if is_media_id(query):
            media = parse_media_id(query)
            if media.source == "local":
                _validate_local_query_path(media.locator)
        else:
            query = query.strip('"').strip("'")
            _validate_local_query_path(query)

    try:
        results = current_app.search_service.semantic_image_search(query, top_k, threshold, media_type)
    except (ValueError, FileNotFoundError) as exc:
        abort(400, str(exc))
    except Exception as exc:
        logger.error("Similar image search failed: %s", exc)
        return jsonify([])
    return jsonify(results)


@main_bp.route("/face_search", methods=["POST"])
def face_search():
    """Search by person name.

    Request JSON:
        ``query`` (str): Person name.

    Returns:
        JSON array of matching image paths.
    """
    data = request.json or {}
    query = data.get("query", "")
    results = current_app.face_service.search_by_name(query)
    return jsonify(results)


@main_bp.route("/integrated_search", methods=["POST"])
def integrated_search():
    """Combined face + activity search.

    Parses queries like ``"find Alice swimming"``.

    Request JSON:
        ``query`` (str): Natural language query.
        ``threshold`` (float, default 0.3): Similarity threshold.
        ``top_k`` (int, default 10): Maximum results.
        ``media_type`` (str, default "image"): Type filter.

    Returns:
        JSON array of intersecting face + semantic results.
    """
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0.3))
    top_k = int(data.get("top_k", 10))
    media_type = data.get("media_type", "image")
    results = current_app.search_service.integrated_face_search(query, top_k, threshold, media_type)
    return jsonify(results)


@main_bp.route("/embed_text", methods=["POST"])
def embed_text():
    """BM25 keyword search over OCR/transcript text.

    Request JSON:
        ``query`` (str): Keyword phrase.
        ``threshold`` (float, default 0.1): Minimum BM25 score.
        ``top_k`` (int, default 5): Maximum results.
        ``media_type`` (str, default "all"): Type filter.

    Returns:
        JSON array of keyword-matched results.
    """
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0.1))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "all")
    results = current_app.search_service.keyword_search(query, top_k, threshold, media_type)
    return jsonify(results)


@main_bp.route("/graph_data", methods=["GET"])
def graph_data():
    """Return the semantic similarity graph.

    Returns:
        JSON object with ``nodes`` and ``links`` arrays.
    """
    try:
        results = current_app.search_service.generate_graph_data()
        return jsonify(results)
    except Exception as exc:
        logger.error("Graph generation failed: %s", exc)
        return jsonify({"nodes": [], "links": []})


@main_bp.route("/subgraph_data", methods=["POST"])
def subgraph_data():
    """Return a subgraph restricted to the given composite IDs.

    The UI calls this after a search when in graph-view mode to show
    only the similarity relationships among the matched results.

    Request JSON:
        ``ids`` (list[str]): Composite IDs of the matched media items.

    Returns:
        JSON object with ``nodes`` and ``links`` arrays.
    """
    data = request.json or {}
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"nodes": [], "links": []})

    try:
        full = current_app.search_service.generate_graph_data()
    except Exception as exc:
        logger.error("Subgraph generation failed: %s", exc)
        return jsonify({"nodes": [], "links": []})

    id_set = set(ids)
    nodes = [n for n in full["nodes"] if n["id"] in id_set]
    node_ids = {n["id"] for n in nodes}
    links = [
        lnk for lnk in full["links"]
        if lnk["source"] in node_ids and lnk["target"] in node_ids
    ]
    return jsonify({"nodes": nodes, "links": links})


# Google Drive OAuth endpoints


@main_bp.route("/integrations/google_drive/status", methods=["GET"])
def google_drive_status():
    """Return the current Google Drive integration status."""
    return jsonify(current_app.google_drive_source.get_status())


@main_bp.route("/integrations/google_drive/auth/start", methods=["POST"])
def google_drive_auth_start():
    """Generate a Google OAuth authorisation URL.

    Returns:
        JSON with ``authorization_url`` and ``state``.
    """
    try:
        payload = current_app.google_drive_source.get_authorization_url()
    except Exception as exc:
        abort(400, str(exc))
    return jsonify(payload)


@main_bp.route("/integrations/google_drive/auth/callback", methods=["GET"])
def google_drive_auth_callback():
    """Handle the OAuth callback and persist credentials.

    Expects ``code`` and ``state`` query parameters.
    """
    code = request.args.get("code")
    state = request.args.get("state")
    if not code:
        abort(400, "Missing authorization code.")
    try:
        current_app.google_drive_source.exchange_code(code, state)
    except Exception as exc:
        abort(400, str(exc))
    return (
        "<html><body><h2>Google Drive connected.</h2>"
        "<p>You can close this window and return to Semantixel.</p></body></html>"
    )


# Static file serving


@main_bp.route("/")
def serve_index():
    """Serve the WebUI landing page."""
    return send_from_directory(current_app.static_folder, "index.html")


@main_bp.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets (CSS, JS, images)."""
    return send_from_directory(os.path.join(current_app.static_folder, "assets"), filename)


@main_bp.route("/images/<path:filename>")
def serve_image(filename):
    """Serve an indexed image file by media ID or path.

    Supports local files and Google Drive image proxying.
    """
    try:
        if is_media_id(filename):
            media = parse_media_id(filename)
            if media.source == "local":
                full_path = media.locator
            elif media.source == current_app.google_drive_source.SOURCE_NAME:
                content, mime_type, file_name = current_app.google_drive_source.fetch_bytes(
                    media.locator
                )
                return send_file(
                    io.BytesIO(content),
                    mimetype=mime_type,
                    download_name=file_name,
                )
            else:
                abort(400, "Unsupported media source.")
        else:
            full_path = describe_local_media(filename).locator
    except ValueError:
        abort(400, "Invalid media identifier.")

    if not is_safe_path(full_path, config.include_directories):
        logger.warning("Unauthorized image access attempt: %s", full_path)
        abort(403, "Access Forbidden")

    directory = os.path.dirname(full_path)
    basename = os.path.basename(full_path)
    return send_from_directory(directory, basename)


# Legacy aliases


@main_bp.route("/ebmed_text", methods=["POST"])
def legacy_embed_text():
    """Legacy endpoint — delegates to :func:`embed_text`."""
    return embed_text()
