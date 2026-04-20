import torch
import torch.nn.functional as F
import time
import os
import io
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from PIL import Image

from semantixel.media import parse_media_id
from semantixel.services.model_manager import model_manager
from semantixel.services.index_service import IndexService
from semantixel.services.face_service import FaceService
from semantixel.core.logging import logger

class SearchService:
    """
    Core search logic for semantic retrieval and graph generation.
    """
    
    def __init__(self, index_service: IndexService, face_service: FaceService):
        self.index_service = index_service
        self.face_service = face_service
        self.image_collection = index_service.image_collection
        self.bm25_service = index_service.bm25_service

    def _process_item_id(self, item_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format an ID to JSON results."""
        if metadata and (
            metadata.get("locator")
            or metadata.get("display_path")
            or metadata.get("source_media_id")
            or metadata.get("source")
        ):
            result = {
                "media_id": metadata.get("source_media_id") or item_id,
                "source": metadata.get("source", "local"),
                "path": metadata.get("display_path") or metadata.get("locator") or item_id,
                "display_path": metadata.get("display_path") or metadata.get("locator") or item_id,
                "type": "video" if metadata.get("type") == "video_frame" else metadata.get("type", "image"),
                "locator": metadata.get("locator", item_id),
                "composite_id": item_id,
            }
            if metadata.get("timestamp") is not None:
                result["timestamp"] = float(metadata["timestamp"])
            return result

        return parse_media_id(item_id).to_result()

    def _is_remote_url(self, query: str) -> bool:
        parsed = urlparse(query)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _fetch_remote_image(self, url: str) -> Image.Image:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        try:
            response = requests.get(url, headers=headers, timeout=(10, 60))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Unable to fetch remote image: {exc}") from exc

        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"Remote URL did not return an image. Content-Type: {content_type}")

        try:
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except OSError as exc:
            raise ValueError("Remote URL response could not be decoded as an image.") from exc

    def _resolve_query_media(self, query: str):
        if self._is_remote_url(query):
            return None, self._fetch_remote_image(query)

        media = parse_media_id(query)
        if media.source == "local":
            return media, media.locator
        if media.source == self.index_service.google_drive_source.SOURCE_NAME:
            return media, self.index_service.google_drive_source.fetch_image(media.locator)
        raise ValueError(f"Unsupported query media source: {media.source}")

    def semantic_text_search(self, query: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "image") -> List[Dict[str, Any]]:
        """
        Performs semantic search on images using CLIP text embeddings.
        """
        logger.info(f"Semantic Text Search: {query} (top_k={top_k}, type={media_type})")
        text_embedding = model_manager.clip.get_text_embeddings(query)
        
        # Increase pool for deduplication
        query_k = top_k * 10
        results = self.image_collection.query(
            query_embeddings=[text_embedding],
            n_results=query_k,
            include=["distances", "metadatas"],
        )
        
        return self._filter_results(results, top_k, threshold, media_type)

    def semantic_image_search(self, image_path: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "all") -> List[Dict[str, Any]]:
        """CLIP Image similarity search."""
        query_media, query_input = self._resolve_query_media(image_path)
        embedding = model_manager.clip.get_image_embeddings([query_input])[0]
        
        query_k = top_k * 10
        results = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=query_k,
            include=["distances", "metadatas"],
        )
        
        # Exclude self if top result is remarkably similar
        exclude_path = query_media.media_id if query_media is not None else None
        return self._filter_results(results, top_k, threshold, media_type, exclude_path=exclude_path)

    def keyword_search(self, query: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "all") -> List[Dict[str, Any]]:
        """BM25 search."""
        ids = self.bm25_service.search(query, top_k, threshold, media_type)
        if not ids:
            return []

        metadata_lookup = {}
        try:
            collection_data = self.image_collection.get(ids=ids, include=["metadatas"])
            for item_id, metadata in zip(collection_data["ids"], collection_data.get("metadatas") or []):
                metadata_lookup[item_id] = metadata
        except Exception:
            metadata_lookup = {}

        return [self._process_item_id(item_id, metadata_lookup.get(item_id)) for item_id in ids]

    def _filter_results(self, results: Dict[str, Any], top_k: int, threshold: float, media_type: str, exclude_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filtering and deduplication (limit frames per video)."""
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0]
        if len(metadatas) != len(ids):
            metadatas = [None] * len(ids)
        
        similarities = [1 - d for d in distances]
        final_results = []
        video_counts = {}
        MAX_FRAMES_PER_VIDEO = 1
        
        for item_id, s, metadata in zip(ids, similarities, metadatas):
            if s <= threshold:
                continue
                
            if exclude_path and item_id == exclude_path:
                continue
                
            item = self._process_item_id(item_id, metadata)
            is_video = item["type"] == "video"
            if media_type == "image" and is_video:
                continue
            if media_type == "video" and not is_video:
                continue
                
            if is_video:
                base_video_path = item["media_id"]
                count = video_counts.get(base_video_path, 0)
                if count >= MAX_FRAMES_PER_VIDEO:
                    continue
                video_counts[base_video_path] = count + 1
            
            final_results.append(item)
            
            if len(final_results) >= top_k:
                break
                
        return final_results

    def generate_graph_data(self) -> Dict[str, Any]:
        """Calculates pairwise cosine similarity and creates node-link graph data."""
        t0 = time.time()
        data = self.image_collection.get(include=["embeddings", "metadatas"])
        ids = data["ids"]
        embeddings = data["embeddings"]
        
        if not ids:
            return {"nodes": [], "links": []}
            
        nodes = []
        metadatas = data.get("metadatas") or [None] * len(ids)
        for doc_id, metadata in zip(ids, metadatas):
            item = self._process_item_id(doc_id, metadata)
            nodes.append({
                "id": doc_id,
                **item,
                "fileName": os.path.basename(item["path"])
            })
            
        # Use PyTorch for efficiency
        embs_tensor = torch.tensor(embeddings)
        sim_matrix = F.cosine_similarity(embs_tensor.unsqueeze(1), embs_tensor.unsqueeze(0), dim=2)
        
        links = []
        TOP_K_NEIGHBORS = 3
        MIN_SIMILARITY = 0.5
        
        sim_matrix.fill_diagonal_(-1.0)
        top_values, top_indices = torch.topk(sim_matrix, min(TOP_K_NEIGHBORS, len(ids) - 1), dim=1)
        
        seen_edges = set()
        for i, source_id in enumerate(ids):
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
        
        logger.info(f"Generated Semantic Graph: {len(nodes)} nodes, {len(links)} edges in {time.time()-t0:.3f}s")
        return {"nodes": nodes, "links": links}

    def integrated_face_search(self, query: str, top_k: int = 10, threshold: float = 0.3, media_type: str = "image") -> List[Dict[str, Any]]:
        """
        Combines face recognition by name and semantic context search.
        Query format example: "Find Karunya playing cricket"
        """
        import re
        query_lower = query.lower().strip()
        
        # Simple regex for "Find [name] [activity]"
        name, activity = None, None
        match = re.search(r'find\s+(\w+)\s+(.+)', query_lower)
        if match:
            name, activity = match.group(1), match.group(2)
        else:
            match = re.search(r'find\s+(\w+)', query_lower)
            if match:
                name = match.group(1)
            else:
                name = query_lower # Fallback
                
        # 1. Face Search
        face_paths = self.face_service.search_by_name(name)
        if not face_paths:
            return []
            
        if not activity:
            return [self._process_item_id(p) for p in face_paths[:top_k]]
            
        # 2. Semantic Search Filter
        semantic_results = self.semantic_text_search(activity, top_k=top_k * 5, threshold=threshold, media_type=media_type)
        semantic_paths = {r["path"] for r in semantic_results}
        
        # 3. Intersection
        final_ids = [p for p in face_paths if p in semantic_paths]
        return [self._process_item_id(p) for p in final_ids[:top_k]]
