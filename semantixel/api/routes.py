import os
from flask import Blueprint, request, jsonify, send_from_directory, current_app, abort
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.core.security import is_safe_path, is_safe_url

main_bp = Blueprint("main", __name__)

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
        # Path validation if it's a local file
        query = query.strip('"').strip("'")
        if not is_safe_path(query, config.include_directories):
            logger.warning(f"Path traversal attempt blocked: {query}")
            abort(403, "Access to this path is forbidden.")
            
    results = current_app.search_service.semantic_image_search(query, top_k, threshold, media_type)
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

@main_bp.route("/")
def serve_index():
    return send_from_directory(current_app.static_folder, "index.html")

@main_bp.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(current_app.static_folder, "assets"), filename)

@main_bp.route("/images/<path:filename>")
def serve_image(filename):
    """
    Secure image serving. Only allows files from included directories.
    """
    # Fix the filename joining to be safe
    # The original was os.path.join("/", filename) which is very unsafe
    full_path = os.path.abspath(os.path.join("/", filename)) if os.name != 'nt' else os.path.abspath(filename)
    
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
