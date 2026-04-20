import io
import os
from flask import Blueprint, request, jsonify, send_file, send_from_directory, current_app, abort
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.core.security import is_safe_path, is_safe_url
from semantixel.media import describe_local_media, is_media_id, parse_media_id

main_bp = Blueprint("main", __name__)


def _validate_local_query_path(query: str) -> str:
    query_media = describe_local_media(query)
    if not is_safe_path(query_media.locator, config.include_directories):
        logger.warning(f"Path traversal attempt blocked: {query}")
        abort(403, "Access to this path is forbidden.")
    return query_media.locator

@main_bp.route("/clip_text", methods=["POST"])
def clip_text():
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "image")
    
    results = current_app.search_service.semantic_text_search(query, top_k, threshold, media_type)
    return jsonify(results)

@main_bp.route("/clip_image", methods=["POST"])
def clip_image():
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "all")
    
    # URL validation for safety
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
    except ValueError as exc:
        abort(400, str(exc))
    return jsonify(results)

@main_bp.route("/face_search", methods=["POST"])
def face_search():
    data = request.json or {}
    query = data.get("query", "")
    results = current_app.face_service.search_by_name(query)
    return jsonify(results)

@main_bp.route("/integrated_search", methods=["POST"])
def integrated_search():
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0.3))
    top_k = int(data.get("top_k", 10))
    media_type = data.get("media_type", "image")
    results = current_app.search_service.integrated_face_search(query, top_k, threshold, media_type)
    return jsonify(results)

@main_bp.route("/embed_text", methods=["POST"])
def embed_text():
    # Mapping the typo'd 'ebmed_text' to the correct 'embed_text' but supporting both if needed
    data = request.json or {}
    query = data.get("query", "")
    threshold = float(data.get("threshold", 0.1))
    top_k = int(data.get("top_k", 5))
    media_type = data.get("media_type", "all")
    
    results = current_app.search_service.keyword_search(query, top_k, threshold, media_type)
    return jsonify(results)

@main_bp.route("/graph_data", methods=["GET"])
def graph_data():
    results = current_app.search_service.generate_graph_data()
    return jsonify(results)

@main_bp.route("/integrations/google_drive/status", methods=["GET"])
def google_drive_status():
    return jsonify(current_app.google_drive_source.get_status())

@main_bp.route("/integrations/google_drive/auth/start", methods=["POST"])
def google_drive_auth_start():
    try:
        payload = current_app.google_drive_source.get_authorization_url()
    except Exception as exc:
        abort(400, str(exc))
    return jsonify(payload)

@main_bp.route("/integrations/google_drive/auth/callback", methods=["GET"])
def google_drive_auth_callback():
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

@main_bp.route("/")
def serve_index():
    return send_from_directory(current_app.static_folder, "index.html")

@main_bp.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(current_app.static_folder, "assets"), filename)

@main_bp.route("/images/<path:filename>")
def serve_image(filename):
    """
    Secure image serving. Resolves local media IDs and only allows files from included directories.
    """
    try:
        if is_media_id(filename):
            media = parse_media_id(filename)
            if media.source == "local":
                full_path = media.locator
            elif media.source == current_app.google_drive_source.SOURCE_NAME:
                content, mime_type, file_name = current_app.google_drive_source.fetch_bytes(media.locator)
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
        logger.warning(f"Unauthorized image access attempt: {full_path}")
        abort(403, "Access Forbidden")
        
    directory = os.path.dirname(full_path)
    basename = os.path.basename(full_path)
    return send_from_directory(directory, basename)

# Handle legacy routes or typos
@main_bp.route("/ebmed_text", methods=["POST"])
def legacy_embed_text():
    return embed_text()
