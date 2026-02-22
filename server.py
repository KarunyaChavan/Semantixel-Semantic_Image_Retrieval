from flask import Flask, abort, request, jsonify, send_from_directory, send_file, render_template_string
from flask_cors import CORS
from Index.create_db import (
    create_vectordb,
    get_clip_image,
    get_clip_text,
    get_text_embeddings,
)
from text_embeddings.bm25_search import BM25TextIndex

import os
import warnings
import logging
import requests
from face_recognition.face_search import search_face_by_name
from face_recognition.integrated_search import integrated_face_semantic_search
from io import BytesIO
import json
import time
from urllib.parse import unquote
import torch
import torch.nn.functional as F

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress specific ChromaDB logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.CRITICAL)

# Setup logging (assuming log_config.py exists and setup_logging is defined)
# If log_config.py is not available, this line will cause an error.
# For now, we'll use a basic logger.
# from log_config import setup_logging
# logger = setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


image_collection, text_collection = create_vectordb("db")
bm25_index = BM25TextIndex()  # Load BM25 index for keyword search


def parse_image(image_path, top_k=5, threshold=0):
    """
    Parses an image from a given path or URL.

    If the image_path is a URL (starts with 'http://' or 'https://'), the function fetches the image
    from the web and returns a BytesIO object containing the image data. If the image_path is a local
    file path, it simply returns the path as is.

    Parameters:
    - image_path (str): The path or URL to the image.

    Returns:
    - BytesIO or str: A BytesIO object containing the image data if the image_path is a URL,
                      or the image_path itself if it's a local file path.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        try:
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch image from URL {image_path}: {e}")
            raise ValueError(f"Failed to fetch image from URL: {e}")
    else:
        image_path = image_path.strip('"').strip("'")
        return image_path


def search_clip_text(text, image_collection, top_k=5, threshold=0, media_type="all"):
    """
    Search for images that are semantically similar to the input text.

    Args:
        text (str): The input text to search for.
        image_collection: The collection of images to search in.
        media_type (str): Filter by media type ('all', 'image', 'video').

    Returns:
        tuple: A tuple containing the paths of the top 5 images and their distances from the input text.
    """
    text_embedding = get_clip_text(text)
    
    # Increase query pool significantly to allow for deduplication of repeated video frames
    query_k = top_k * 10
    
    results = image_collection.query(
        query_embeddings=text_embedding, 
        n_results=query_k
    )
    
    similarities = [1 - d for d in results["distances"][0]]
    paths = []
    final_similarities = []
    video_counts = {}
    MAX_FRAMES_PER_VIDEO = 1
    
    for p, d in zip(results["ids"][0], similarities):
        if d > threshold:
            # Apply media_type filter based on composite ID pattern
            is_video = ":::" in p
            if media_type == "image" and is_video:
                continue
            if media_type == "video" and not is_video:
                continue
                
            # NMS/Deduplication for videos: Limit frames per identical source video
            if is_video:
                base_video_path = p.split(":::")[0]
                current_count = video_counts.get(base_video_path, 0)
                if current_count >= MAX_FRAMES_PER_VIDEO:
                    continue  # Skip redundant frame
                video_counts[base_video_path] = current_count + 1
                
            paths.append(p)
            final_similarities.append(d)
            
            if len(paths) >= top_k:
                break
                
    return paths, final_similarities


def search_clip_image(
    image_path, image_collection, top_k=5, threshold=0, get_self=False, media_type="all"
):
    """
    Search for images that are visually similar to the input image within a given image collection.

    Args:
        image_path (str): The path to the input image to search for. This path is stripped of any leading or trailing quotes and adjusted for posix systems.
        image_collection (FaissCollection): The collection of images to search in. This is an object that supports querying for nearest neighbors.
        get_self (bool, optional): If set to True, the function will return the input image as one of the results.
        media_type (str): Filter by media type ('all', 'image', 'video').
    Returns:
        tuple: A tuple containing two lists. The first list contains the paths of the top 5 images (or top 6 if get_self is True). The second list contains the corresponding distances of these images from the input image.
    """
    image_embedding = get_clip_image([image_path])
    
    # Increase query pool significantly to allow for deduplication of repeated video frames
    # Add 1 to account for potentially filtering out the self-image
    query_k = (top_k * 10) + 1
        
    results = image_collection.query(
        query_embeddings=image_embedding, 
        n_results=query_k
    )
    
    similarities = [1 - d for d in results["distances"][0]]
    paths = []
    final_similarities = []
    video_counts = {}
    MAX_FRAMES_PER_VIDEO = 1
    
    for p, d in zip(results["ids"][0], similarities):
        if d > threshold:
            # Exclude self if requested
            if not get_self and p == image_path:
                continue
                
            # Apply media_type filter based on composite ID pattern
            is_video = ":::" in p
            if media_type == "image" and is_video:
                continue
            if media_type == "video" and not is_video:
                continue
                
            # NMS/Deduplication for videos: Limit frames per identical source video
            if is_video:
                base_video_path = p.split(":::")[0]
                current_count = video_counts.get(base_video_path, 0)
                if current_count >= MAX_FRAMES_PER_VIDEO:
                    continue  # Skip redundant frame
                video_counts[base_video_path] = current_count + 1
                
            paths.append(p)
            final_similarities.append(d)
            
            if len(paths) >= top_k:
                break
                
    return paths, final_similarities


def search_embed_text(text, text_collection, top_k=5, threshold=0):
    """
    Search for texts that are semantically similar to the input text.

    Args:
        text (str): The input text to search for.
        text_collection: The collection of texts to search in.

    Returns:
        tuple: A tuple containing the paths of the top 5 texts and their distances from the input text.
    """
    text_embedding = get_text_embeddings(text)
    results = text_collection.query(text_embedding, n_results=top_k)
    similarities = [1 - d for d in results["distances"][0]]
    paths, similarities = [
        p for p, d in zip(results["ids"][0], similarities) if d > threshold
    ], [d for d in similarities if d > threshold]
    return paths, similarities

def process_search_results(paths):
    """Helper function to format search results."""
    formatted_results = []
    for path in paths:
        if ":::" in path:
            video_path, timestamp = path.split(":::")
            formatted_results.append({
                "path": video_path,
                "type": "video",
                "timestamp": float(timestamp),
                "composite_id": path
            })
        else:
            formatted_results.append({
                "path": path,
                "type": "image"
            })
    return formatted_results

# Flask App
app = Flask(__name__, static_folder="UI/Semantixel WebUI")
CORS(app)


@app.route("/clip_text", methods=["POST"])
def clip_text_route():
    """
    Handle a POST request to search images via text queries (using CLIP).

    Retrieves the following JSON fields from the request:
        - query (str): The text query to search for.
        - threshold (float): The minimum similarity threshold. Defaults to 0.
        - top_k (int): The number of top results to return. Defaults to 5.

    Calls `search_clip_text` with these parameters to retrieve a list of image
    paths (and their associated distances). Returns the list of image paths as JSON.

    Returns:
        flask.Response: A JSON response containing a list of image paths.
    """
    query = request.json.get("query", "")
    threshold = float(request.json.get("threshold", 0))
    top_k = int(request.json.get("top_k", 5))
    media_type = request.json.get("media_type", "all")
    print(f"threshold: {threshold} top_k: {top_k} media_type: {media_type}")
    paths, distances = search_clip_text(query, image_collection, top_k, threshold, media_type)
    print(len(paths))
    
    # Format paths to distinguish between images and video frames
    formatted_results = []
    for path in paths:
        if ":::" in path:
            video_path, timestamp = path.split(":::")
            formatted_results.append({
                "path": video_path,
                "type": "video",
                "timestamp": float(timestamp),
                "composite_id": path
            })
        else:
            formatted_results.append({
                "path": path,
                "type": "image"
            })
            
    return jsonify(formatted_results)


@app.route("/clip_image", methods=["POST"])
def clip_image_route():
    """
    Handle a POST request to search images via an image query (using CLIP).

    Retrieves the following JSON fields from the request:
        - query (str): Base64-encoded or URL reference to the image.
        - threshold (float): The minimum similarity threshold. Defaults to 0.
        - top_k (int): The number of top results to return. Defaults to 5.

    Calls `parse_image` to transform the input into a usable format, then uses
    `search_clip_image` to find matching images in the collection. Returns the
    list of matching image paths as JSON.

    Returns:
        flask.Response: A JSON response containing a list of image paths.
    """
    query = request.json.get("query", "")
    threshold = float(request.json.get("threshold", 0))
    top_k = int(request.json.get("top_k", 5))
    media_type = request.json.get("media_type", "all")
    query = parse_image(query)
    paths, distances = search_clip_image(query, image_collection, top_k, threshold, False, media_type)
    
    # Format paths
    formatted_results = []
    for path in paths:
        if ":::" in path:
            video_path, timestamp = path.split(":::")
            formatted_results.append({
                "path": video_path,
                "type": "video",
                "timestamp": float(timestamp),
                "composite_id": path
            })
        else:
            formatted_results.append({
                "path": path,
                "type": "image"
            })
            
    return jsonify(formatted_results)


@app.route("/ebmed_text", methods=["POST"])
def ebmed_text_route():
    """
    Handle a POST request to search text embeddings using BM25 keyword search.

    Retrieves the following JSON fields from the request:
        - query (str): The keyword(s) to search for in OCR content.
        - threshold (float): Minimum BM25 score threshold. Defaults to 0.
        - top_k (int): The number of top results to return. Defaults to 5.

    Uses BM25 algorithm for lexical/keyword matching instead of semantic search,
    which is more suitable for exact text finding in OCR content.

    Returns:
        flask.Response: A JSON response containing a list of image paths with matching text.
    """
    try:
        data = request.json
        query = data.get("query", "")
        threshold = float(data.get("threshold", 0.1))
        top_k = int(data.get("top_k", 5))
        media_type = data.get("media_type", "all")
    
        if not query:
            return jsonify([])
        
        # Use BM25 keyword search instead of semantic embeddings
        paths = bm25_index.search(query, top_k=top_k, threshold=threshold, media_type=media_type)
        
        # Format paths
        formatted_results = []
        for path in paths:
            if ":::" in path:
                video_path, timestamp = path.split(":::")
                formatted_results.append({
                    "path": video_path,
                    "type": "video",
                    "timestamp": float(timestamp),
                    "composite_id": path
                })
            else:
                formatted_results.append({
                    "path": path,
                    "type": "image"
                })
                
        return jsonify(formatted_results)
    except Exception as e:
        logger.error(f"Error in ebmed_text route: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/face_search", methods=["POST"])
def face_search_route():
    query = request.json.get("query", "")
    results = search_face_by_name(query)
    return jsonify(results)

@app.route("/integrated_search", methods=["POST"])
def integrated_search_route():
    query = request.json.get("query", "")
    threshold = float(request.json.get("threshold", 0.3))
    top_k = int(request.json.get("top_k", 10))
    results = integrated_face_semantic_search(query, image_collection, top_k, threshold)
    return jsonify(results)

@app.route("/graph_data", methods=["GET"])
def graph_data_route():
    """
    Returns a JSON structure containing { nodes: [...], links: [...] }.
    Calculates the pairwise cosine similarity between all embeddings in the DB
    and creates edges for the top K closest neighbors of each node.
    """
    try:
        t0 = time.time()
        
        # 1. Fetch all embeddings and metadata
        data = image_collection.get(include=["embeddings"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        
        if not ids or len(ids) == 0:
            return jsonify({"nodes": [], "links": []})
            
        # Create Nodes
        nodes = []
        for i, doc_id in enumerate(ids):
            is_video = ":::" in doc_id
            
            if is_video:
                path, timestamp = doc_id.rsplit(":::", 1)
                file_name = os.path.basename(path)
                nodes.append({
                    "id": doc_id,
                    "composite_id": doc_id,
                    "path": path,
                    "type": "video",
                    "timestamp": float(timestamp),
                    "fileName": f"{file_name} ({timestamp}s)"
                })
            else:
                nodes.append({
                    "id": doc_id,
                    "composite_id": doc_id,
                    "path": doc_id,
                    "type": "image",
                    "fileName": os.path.basename(doc_id)
                })
                
        # 2. Calculate similarities efficiently using PyTorch
        embs_tensor = torch.tensor(embeddings)
        # Cosine similarity matrix (N x N)
        sim_matrix = F.cosine_similarity(embs_tensor.unsqueeze(1), embs_tensor.unsqueeze(0), dim=2)
        
        # 3. Create Links (Sparse Graph)
        # For a clean visual graph, we only connect each node to its top ~3 closest neighbors
        TOP_K_NEIGHBORS = 3
        MIN_SIMILARITY = 0.5 # Ignore incredibly weak links even if they are in the top 3
        
        links = []
        
        # Prevent self-loops (diagonal = 1.0) by filling diagonal with -1
        sim_matrix.fill_diagonal_(-1.0)
        
        # Get top K values and indices for each row
        top_values, top_indices = torch.topk(sim_matrix, min(TOP_K_NEIGHBORS, len(ids) - 1), dim=1)
        
        # Keep track of added edges to prevent bidirectional duplicates like A->B and B->A
        seen_edges = set()
        
        for i in range(len(ids)):
            source_id = ids[i]
            for j in range(top_indices.shape[1]):
                target_idx = top_indices[i, j].item()
                similarity = top_values[i, j].item()
                target_id = ids[target_idx]
                
                if similarity > MIN_SIMILARITY:
                    # Create undirected edge key
                    edge_tuple = tuple(sorted([source_id, target_id]))
                    if edge_tuple not in seen_edges:
                        seen_edges.add(edge_tuple)
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "value": float(similarity)
                        })
                        
        logger.info(f"Generated Semantic Graph: {len(nodes)} nodes, {len(links)} edges in {time.time()-t0:.3f}s")
        return jsonify({
            "nodes": nodes,
            "links": links
        })
        
    except Exception as e:
        logger.error(f"Error generating graph data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/subgraph_data", methods=["POST"])
def subgraph_data_route():
    """
    Returns a Graph structure for a specific subset of composite IDs (Search Results).
    """
    try:
        t0 = time.time()
        data_req = request.json
        composite_ids = data_req.get("ids", [])
        
        if not composite_ids:
            return jsonify({"nodes": [], "links": []})
            
        # Fetch embeddings only for the provided IDs
        data = image_collection.get(ids=composite_ids, include=["embeddings"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        
        if not ids or len(ids) == 0:
            return jsonify({"nodes": [], "links": []})
            
        nodes = []
        for i, doc_id in enumerate(ids):
            is_video = ":::" in doc_id
            if is_video:
                path, timestamp = doc_id.rsplit(":::", 1)
                file_name = os.path.basename(path)
                nodes.append({
                    "id": doc_id,
                    "composite_id": doc_id,
                    "path": path,
                    "type": "video",
                    "timestamp": float(timestamp),
                    "fileName": f"{file_name} ({timestamp}s)"
                })
            else:
                nodes.append({
                    "id": doc_id,
                    "composite_id": doc_id,
                    "path": doc_id,
                    "type": "image",
                    "fileName": os.path.basename(doc_id)
                })
                
        # Calculate similarities 
        embs_tensor = torch.tensor(embeddings)
        sim_matrix = F.cosine_similarity(embs_tensor.unsqueeze(1), embs_tensor.unsqueeze(0), dim=2)
        
        # Sub-graphs are usually small (10-30 nodes). Connect top 2 neighbors to show clusters
        TOP_K_NEIGHBORS = min(2, len(ids) - 1)
        MIN_SIMILARITY = 0.5 
        
        links = []
        if TOP_K_NEIGHBORS > 0:
            sim_matrix.fill_diagonal_(-1.0)
            top_values, top_indices = torch.topk(sim_matrix, TOP_K_NEIGHBORS, dim=1)
            
            seen_edges = set()
            for i in range(len(ids)):
                source_id = ids[i]
                for j in range(top_indices.shape[1]):
                    target_idx = top_indices[i, j].item()
                    similarity = top_values[i, j].item()
                    target_id = ids[target_idx]
                    
                    if similarity > MIN_SIMILARITY:
                        edge_tuple = tuple(sorted([source_id, target_id]))
                        if edge_tuple not in seen_edges:
                            seen_edges.add(edge_tuple)
                            links.append({
                                "source": source_id,
                                "target": target_id,
                                "value": float(similarity)
                            })
                            
        logger.info(f"Generated Semantic Sub-Graph: {len(nodes)} nodes, {len(links)} edges in {time.time()-t0:.3f}s")
        return jsonify({
            "nodes": nodes,
            "links": links
        })
        
    except Exception as e:
        logger.error(f"Error generating subgraph data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def serve_index():
    """
    Serve the main index page (index.html) from the static folder.

    Returns:
        flask.Response: The index.html file from the `app.static_folder`.
    """
    return send_from_directory(app.static_folder, "index.html")

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.static_folder, "assets"), filename)


@app.route("/images/<path:filename>")
def serve_image(filename):
    """
    Serve an image file from within the images directory.

    Args:
        filename (str): The path to the image file within the images directory.

    Returns:
        flask.Response: The requested image file from its directory.
    """
    filename = os.path.join("/", filename)
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 23107))
    app.run(host="0.0.0.0", port=port)
