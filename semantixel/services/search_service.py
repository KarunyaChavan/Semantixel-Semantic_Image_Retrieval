import torch
import torch.nn.functional as F
import time
import os
import io
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse

import requests
from PIL import Image

from semantixel.core.config import config
from semantixel.media import parse_media_id
from semantixel.services.model_manager import model_manager
from semantixel.services.index_service import IndexService
from semantixel.services.face_service import FaceService
from semantixel.core.logging import logger

class SearchService:
    """Aggregated search across image, text, and audio modalities.

    Each modality produces cosine distances in a different characteristic
    range. ``MODALITY_RANGES`` stores the typical similarity interval
    (``[min_s, max_s]``) for relevant results per modality. The
    :meth:`_normalize_distance` helper maps raw cosine distances into
    comparable ``[0, 1]`` scores so results from CLIP, MiniLM, and CLAP
    can be merged and ranked fairly in a single list.

    ``MODALITY_RANGES`` keys:

    * ``clip``   — visual / text CLIP embeddings (dim 512)
    * ``minilm`` — sentence-transformer MiniLM  (dim 384)
    * ``clap``   — CLAP audio / text embeddings (dim 512)
    """

    MODALITY_RANGES = {
        "clip":   {"min_s": 0.10, "max_s": 0.35},
        "minilm": {"min_s": 0.15, "max_s": 0.75},
        "clap":   {"min_s": 0.10, "max_s": 0.30},
    }

    def __init__(self, index_service: IndexService, face_service: FaceService):
        self.index_service = index_service
        self.face_service = face_service
        self.image_collection = index_service.image_collection
        self.text_collection = index_service.text_collection
        self.audio_collection = index_service.audio_collection
        self.bm25_service = index_service.bm25_service

        self._modalities: List[tuple[Callable, Any, str]] = [
            (model_manager.clip.get_text_embeddings, self.image_collection, "clip"),
            (model_manager.text_embed.get_embeddings, self.text_collection, "minilm"),
        ]
        if config.audio.clap_enabled:
            self._modalities.append(
                (model_manager.clap.get_text_embeddings, self.audio_collection, "clap")
            )

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

            raw_type = metadata.get("type", "image")
            item_type = "video" if raw_type == "video_frame" else raw_type
            
            result = {
                "media_id": metadata.get("source_media_id") or metadata.get("source_file") or item_id,
                "source": metadata.get("source", "local"),
                "path": path_val,
                "display_path": path_val,
                "type": item_type,
                "locator": metadata.get("locator") or metadata.get("source_file") or item_id,
                "composite_id": item_id,
            }
            if metadata.get("timestamp") is not None:
                result["timestamp"] = float(metadata["timestamp"])
            elif raw_type == "video_frame":
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

    @staticmethod
    def _normalize_distance(d: float, modality: str) -> float:
        """Convert a raw cosine distance into a calibrated ``[0, 1]`` similarity score.

        Raw cosine distances from ChromaDB are modality-dependent:
        CLIP distances cluster around ``[0.10, 0.35]`` while MiniLM
        distances span ``[0.20, 0.75]``. This method maps each modality's
        *typical relevant range* to the unit interval so scores are
        comparable when merging results from different collections.

        The mapping is::

            s = 1.0 - d
            score = clamp((s - min_s) / (max_s - min_s), 0, 1)

        Args:
            d: Raw cosine distance from ChromaDB (0 = identical).
            modality: One of ``"clip"``, ``"minilm"``, ``"clap"``.

        Returns:
            Normalised similarity in ``[0, 1]`` (1 = perfect match).
        """
        s = 1.0 - d
        r = SearchService.MODALITY_RANGES.get(modality, {"min_s": 0.0, "max_s": 1.0})
        if s <= r["min_s"]:
            return 0.0
        if s >= r["max_s"]:
            return 1.0
        return (s - r["min_s"]) / (r["max_s"] - r["min_s"])

    @staticmethod
    def _is_lyrics_query(query: str) -> bool:
        """Heuristic: queries with 3+ words are treated as lyric-like.

        When a multi-word query is detected, transcript matches receive
        a small boost in :meth:`semantic_text_search` because lyric
        phrases are more likely to appear in Whisper transcriptions than
        in CLAP ambient embeddings.

        Args:
            query: The raw user query string.

        Returns:
            ``True`` if the query has at least three whitespace-separated tokens.
        """
        return len(query.strip().split()) >= 3

    def _query_collection(self, embedding_fn: Callable, collection, query: str, query_k: int) -> dict:
        """Encode *query* and run a vector search against *collection*.

        Args:
            embedding_fn: Callable that maps a string to a list of floats.
            collection: ChromaDB collection to search.
            query: Raw text query.
            query_k: Number of nearest neighbours to request.

        Returns:
            ChromaDB result dict with keys ``ids``, ``distances``, ``metadatas``.
        """
        embedding = embedding_fn(query)
        return collection.query(
            query_embeddings=[embedding],
            n_results=query_k,
            include=["distances", "metadatas"]
        )
    
    def semantic_text_search(self, query: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "image") -> List[Dict[str, Any]]:
        """Unified natural-language search across all indexed modalities.

        Queries the image collection (CLIP), text collection (MiniLM),
        and — when enabled — the ambient audio collection (CLAP). Results
        from each collection are normalised with :meth:`_normalize_distance`,
        merged, sorted by descending similarity, deduplicated, and filtered.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (inclusive) for a result to be kept.
            media_type: ``"image"``, ``"video"``, ``"audio"``, or ``"all"``.

        Returns:
            List of result dicts, each containing ``media_id``, ``path``,
            ``type``, ``timestamp`` (if applicable), and ``similarity``.
        """
        logger.info(f"Unified Semantic Search: {query} (top_k={top_k}, type={media_type})")
        query_k = top_k * 10
        is_lyrics = self._is_lyrics_query(query)

        combined_items = []
        for embedding_fn, collection, modality in self._modalities:
            results = self._query_collection(embedding_fn, collection, query, query_k)
            if not results["ids"] or not results["ids"][0]:
                continue
            for p, d, m in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
                s = self._normalize_distance(d, modality)
                if is_lyrics and modality == "minilm":
                    subtype = (m or {}).get("subtype", "")
                    if subtype == "transcript":
                        s = min(1.0, s * 1.25)
                combined_items.append((p, s, m))

        combined_items.sort(key=lambda x: x[1], reverse=True)

        merged_results = {
            "ids": [[item[0] for item in combined_items]],
            "distances": [[item[1] for item in combined_items]],
            "metadatas": [[item[2] for item in combined_items]]
        }

        if not merged_results["ids"] or not merged_results["ids"][0]:
            return []

        return self._filter_results(merged_results, top_k, threshold, media_type)

    def semantic_image_search(self, image_path: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "all") -> List[Dict[str, Any]]:
        """Find visually similar images using CLIP embedding.

        Accepts a local path, Google Drive reference, or remote URL as
        the query image. The query image itself is excluded from results.

        Args:
            image_path: Local path, media ID, or URL of the query image.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (inclusive).
            media_type: ``"image"``, ``"video"``, ``"audio"``, or ``"all"``.

        Returns:
            List of result dicts ordered by descending similarity.
        """
        query_media, query_input = self._resolve_query_media(image_path)
        embedding = model_manager.clip.get_image_embeddings([query_input])[0]

        query_k = top_k * 10
        results = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=query_k,
            include=["distances", "metadatas"],
        )

        exclude_path = query_media.media_id if query_media is not None else None
        results["distances"][0] = [self._normalize_distance(d, "clip") for d in results["distances"][0]]
        return self._filter_results(results, top_k, threshold, media_type, exclude_path=exclude_path)

    def keyword_search(self, query: str, top_k: int = 5, threshold: float = 0.0, media_type: str = "all") -> List[Dict[str, Any]]:
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
        """Apply threshold, type filter, and deduplication to raw ChromaDB results.

        Deduplication logic:

        * **Video items** — at most *MAX_FRAMES_PER_VIDEO* (1) result per
          ``base_video_path`` (the media ID before any ``:::`` postfix).
          This collapses frames, transcripts, and ambient entries from
          the same file into a single result.
        * **Image / audio items** — each ``media_id`` appears at most once.

        Args:
            results: ChromaDB result dict with ``ids``, ``distances``, ``metadatas``.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity (inclusive) to keep a result.
            media_type: ``"image"``, ``"video"``, ``"audio"``, or ``"all"``.
            exclude_path: Optional media ID to exclude (used in image search).

        Returns:
            Filtered, deduplicated result list.
        """
        ids = results["ids"][0]
        similarities = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0]
        if len(metadatas) != len(ids):
            metadatas = [None] * len(ids)
        
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
        import re
        query_lower = query.lower().strip()
        
        name, activity = None, None
        match = re.search(r'find\s+(\w+)\s+(.+)', query_lower)
        if match:
            name, activity = match.group(1), match.group(2)
        else:
            match = re.search(r'find\s+(\w+)', query_lower)
            if match:
                name = match.group(1)
            else:
                name = query_lower
                
        face_paths = self.face_service.search_by_name(name)
        if not face_paths:
            return []
            
        if not activity:
            return [self._process_item_id(p) for p in face_paths[:top_k]]
            
        semantic_results = self.semantic_text_search(activity, top_k=top_k * 5, threshold=threshold, media_type=media_type)
        semantic_paths = {r["path"] for r in semantic_results}
        
        final_ids = [p for p in face_paths if p in semantic_paths]
        return [self._process_item_id(p) for p in final_ids[:top_k]]
