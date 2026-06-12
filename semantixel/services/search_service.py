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
        self.text_collection = index_service.text_collection
        self.audio_collection = index_service.audio_collection
        self.bm25_service = index_service.bm25_service

    def _process_item_id(self, item_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if metadata and (
            metadata.get("locator")
            or metadata.get("display_path")
            or metadata.get("source_media_id")
            or metadata.get("source")
            or metadata.get("source_file")
        ):
            path_val = (
                metadata.get("display_path") 
                or metadata.get("locator") 
                or metadata.get("source_file") 
                or item_id
            )
            
            result = {
                "media_id": metadata.get("source_media_id") or metadata.get("source_file") or item_id,
                "source": metadata.get("source", "local"),
                "path": path_val,
                "display_path": path_val,
                "type": metadata.get("type", "image"),
                "locator": metadata.get("locator") or metadata.get("source_file") or item_id,
                "composite_id": item_id,
            }
            if metadata.get("timestamp") is not None:
                result["timestamp"] = float(metadata["timestamp"])
            elif metadata.get("type") == "video_frame":
                result["timestamp"] = 0.0
            return result

        if ":::" in item_id:
            try:
                base_media_id, postfix = item_id.split(":::", 1)
                if postfix in ("audio", "ambient"):
                    parsed = parse_media_id(base_media_id)
                    is_video = any(parsed.locator.lower().endswith(ext) for ext in [".mp4", ".mkv", ".avi", ".mov"])
                    return {
                        "media_id": base_media_id,
                        "source": parsed.source,
                        "path": parsed.locator,
                        "display_path": parsed.locator,
                        "type": "video" if is_video else "audio",
                        "timestamp": 0.0 if is_video else None,
                        "locator": parsed.locator,
                        "composite_id": item_id
                    }
            except ValueError:
                pass

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

    def _normalize_distance(self, raw_distance: float, model_type: str) -> float:
        """
        Normalizes raw distance values into a 0–1 similarity score
        based on model-specific expected similarity ranges.

        The normalization thresholds are determined empirically by
        analyzing similarity distributions for:
        
            - Relevant matches (true positives / semantically related results)
            - Irrelevant matches (negative or unrelated results)

        Guidelines:
        - `s_min` should represent the lower boundary of typically
           relevant similarity scores.

        - `s_max` should represent the upper boundary of highly
           confident matches.

        This normalization improves consistency across different
        embedding models and retrieval modalities.
        """
        s = 1.0 - raw_distance
        if model_type == "clip":
            s_min, s_max = 0.12, 0.30
        elif model_type == "minilm":
            s_min, s_max = 0.20, 0.70
        elif model_type == "clap":
            s_min, s_max = 0.10, 0.28
        else:
            s_min, s_max = 0.0, 1.0
            
        s_norm = (s - s_min) / (s_max - s_min) if s_max > s_min else 0.0
        s_norm = max(0.0, min(1.0, s_norm))
        return 1.0 - s_norm
    
    def _query_collection(self, embedding_fn, collection, query, query_k):
        """
        Helper to query a ChromaDB collection with the appropriate embedding function.
        """
        embedding = embedding_fn(query)

        return collection.query(
            query_embeddings=[embedding],
            n_results=query_k,
            include=["distances", "metadatas"]
        )
    
    def semantic_text_search(self, query: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "image") -> List[Dict[str, Any]]:
        """
        Performs semantic search across both images and transcribed audio/OCR texts
        """
        logger.info(f"Unified Semantic Search: {query} (top_k={top_k}, type={media_type})")
        query_k = top_k * 10
        
        # 1. Image Collection (CLIP)
        visual_results = self._query_collection(
            model_manager.clip.get_text_embeddings,
            self.image_collection,
            query,
            query_k
        )
        
        # 2. Text Collection (MiniLM)
        text_results = self._query_collection(
            model_manager.text_embed.get_embeddings,
            self.text_collection,
            query,
            query_k
        )
        
        # 3. Ambient Audio Collection (CLAP)
        ambient_results = self._query_collection(
            model_manager.clap.get_text_embeddings,
            self.audio_collection,
            query,
            query_k
        )
        
        # Zip and Merge
        combined_items = []
        if visual_results["ids"] and visual_results["ids"][0]:
            for p, d, m in zip(visual_results["ids"][0], visual_results["distances"][0], visual_results["metadatas"][0]):
                d_norm = self._normalize_distance(d, "clip")
                combined_items.append((p, d_norm, m))
                
        if text_results["ids"] and text_results["ids"][0]:
            for p, d, m in zip(text_results["ids"][0], text_results["distances"][0], text_results["metadatas"][0]):
                d_norm = self._normalize_distance(d, "minilm")
                combined_items.append((p, d_norm, m))
                
        if ambient_results["ids"] and ambient_results["ids"][0]:
            for p, d, m in zip(ambient_results["ids"][0], ambient_results["distances"][0], ambient_results["metadatas"][0]):
                d_norm = self._normalize_distance(d, "clap")
                combined_items.append((p, d_norm, m))
                
        # Sort combined by distance ascending (lower distance = higher similarity)
        combined_items.sort(key=lambda x: x[1])
        
        # Re-pack into ChromaDB native format for the filter function
        merged_results = {
            "ids": [[item[0] for item in combined_items]],
            "distances": [[item[1] for item in combined_items]],
            "metadatas": [[item[2] for item in combined_items]]
        }
        
        if not merged_results["ids"] or not merged_results["ids"][0]:
            return []
            
        return self._filter_results(merged_results, top_k, threshold, media_type)

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
        seen_media_ids = set()
        MAX_FRAMES_PER_VIDEO = 1
        
        for item_id, s, metadata in zip(ids, similarities, metadatas):
            if s <= threshold:
                continue
                
            if exclude_path and item_id == exclude_path:
                continue
                
            item_info = self._process_item_id(item_id, metadata)
            item_type = item_info["type"]
            
            if media_type != "all" and media_type != item_type:
                continue
                
            if item_type == "video":
                base_video_path = item_id.split(":::")[0]
                count = video_counts.get(base_video_path, 0)
                if count >= MAX_FRAMES_PER_VIDEO:
                    continue
                video_counts[base_video_path] = count + 1
            else:
                if item_info["media_id"] in seen_media_ids:
                    continue
                seen_media_ids.add(item_info["media_id"])
            
            final_results.append(item_info)
            
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
