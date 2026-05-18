import os
import chromadb
from tqdm import tqdm
from typing import List
from semantixel.core.config import config
from semantixel.core.logging import logger
from semantixel.media import MediaDescriptor, describe_local_media
from semantixel.sources import GoogleDriveSource
from semantixel.services.model_manager import model_manager
from semantixel.services.bm25_service import BM25Service
from semantixel.utils.scan_utils import fast_scan_for_media
from semantixel.utils.video_utils import extract_frames_in_memory

class IndexService:
    """
    Core service for managing the media index (VectorDB + Keyword search).
    """
    
    def __init__(self, db_path: str = "db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.image_collection = self.client.get_or_create_collection(
            "images", metadata={"hnsw:space": "cosine"}
        )
        self.text_collection = self.client.get_or_create_collection(
            "texts", metadata={"hnsw:space": "cosine"}
        )
        self.audio_collection = self.client.get_or_create_collection(
            "ambient_audio", metadata={"hnsw:space": "cosine"}
        )
        self.bm25_service = BM25Service(index_path=os.path.join(db_path, "bm25_index.pkl"))
        self.video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
        self.audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
        self.google_drive_source = GoogleDriveSource()

    def run_full_scan(self):
        """
        Scans directories and indexes all found media.
        """
        logger.info("Starting full media scan and index update")
        include_dirs = config.include_directories
        exclude_dirs = config.exclude_directories
        
        if not include_dirs:
            logger.warning("No include_directories configured. Skipping scan.")
            return

        paths, elapsed = fast_scan_for_media(include_dirs, exclude_dirs)
        media_items = [describe_local_media(path) for path in paths]

        if self.google_drive_source.is_enabled():
            try:
                media_items.extend(self.google_drive_source.list_media())
            except Exception as exc:
                logger.warning(f"Google Drive scan skipped: {exc}")

        logger.info(f"Found {len(media_items)} media files in {elapsed:.2f}s")
        
        self.index_media(media_items)
        self.cleanup_index(media_items)

    def _get_processing_input(self, media: MediaDescriptor):
        if media.source == "local":
            return media.locator
        if media.source == self.google_drive_source.SOURCE_NAME:
            return self.google_drive_source.fetch_image(media.locator)
        raise ValueError(f"Unsupported media source: {media.source}")

    def index_media(self, media_items: List[MediaDescriptor]):
        """
        Processes and indexes the provided media items.
        """
        batch_size = config.batch_size
        deep_scan = config.deep_scan
        
        # Priority De-coupling: Index fast visuals instantly, reserve audio indexing for very end
        audio_items = [m for m in media_items if m.locator.lower().endswith(tuple(self.audio_extensions | self.video_extensions))]
        visual_items = [m for m in media_items if not m.locator.lower().endswith(tuple(self.audio_extensions))]
        
        total_tasks = len(visual_items) + len(audio_items)
        with tqdm(total=total_tasks, desc="Indexing media") as pbar:
            processing_inputs = []
            processing_ids = []
            processing_metadatas = []
            
            def flush_batch():
                if not processing_inputs:
                    return
                
                logger.debug(f"Flushing batch of {len(processing_inputs)} items")
                
                # Visual Embeddings
                image_embeddings = model_manager.clip.get_image_embeddings(processing_inputs)
                self.image_collection.upsert(
                    ids=processing_ids,
                    embeddings=image_embeddings,
                    metadatas=processing_metadatas
                )
                
                # OCR & Text Embeddings
                ocr_texts = model_manager.ocr.apply_ocr(processing_inputs)
                
                for idx, text in enumerate(ocr_texts):
                    if text:
                        current_id = processing_ids[idx]
                        metadata = processing_metadatas[idx]
                        
                        # Semantic Text Search
                        text_embedding = model_manager.text_embed.get_embeddings(text)
                        self.text_collection.upsert(
                            ids=[current_id],
                            embeddings=[text_embedding],
                            metadatas=[metadata]
                        )
                        
                        # Keyword Search
                        self.bm25_service.add_document(current_id, text)
                
                processing_inputs.clear()
                processing_ids.clear()
                processing_metadatas.clear()

            # PHASE 1: Process Visuals
            for media in visual_items:
                path = media.locator
                is_video = path.lower().endswith(tuple(self.video_extensions))
                needs_indexing = False
                
                # Check if already indexed
                if is_video:
                    results = self.image_collection.get(where={"source_media_id": media.media_id})
                    if not results["ids"]:
                        needs_indexing = True
                else:
                    results = self.image_collection.get(ids=[media.media_id])
                    if not results["ids"]:
                        needs_indexing = True
                    elif deep_scan:
                        # Deep scan logic (e.g., checksum or average) - for now just check existence
                        # To be fully production-ready, we should hash the file
                        pass

                if needs_indexing:
                    if is_video:
                        for frame in extract_frames_in_memory(path):
                            processing_inputs.append(frame["image"])
                            frame_media = describe_local_media(path, timestamp=frame["timestamp"])
                            processing_ids.append(frame_media.composite_id)
                            processing_metadatas.append({
                                "source": frame_media.source,
                                "source_media_id": frame_media.media_id,
                                "locator": frame_media.locator,
                                "display_path": frame_media.display_path,
                                "timestamp": frame["timestamp"],
                                "type": "video_frame",
                            })
                            
                            if len(processing_inputs) >= batch_size:
                                flush_batch()
                    else:
                        processing_inputs.append(self._get_processing_input(media))
                        processing_ids.append(media.media_id)
                        processing_metadatas.append({
                            "source": media.source,
                            "source_media_id": media.media_id,
                            "locator": media.locator,
                            "display_path": media.display_path,
                            "type": "image",
                        })
                        
                        if len(processing_inputs) >= batch_size:
                            flush_batch()
                
                pbar.update(1)
            
            flush_batch() # Last one for visual
            
            # PHASE 2: Process Audio Constraints Sequentially
            for media in audio_items:
                # 1. Transcript indexing
                transcript_id = f"{media.media_id}:::audio"
                transcript_results = self.text_collection.get(ids=[transcript_id])
                
                if not transcript_results["ids"]:
                    transcript = model_manager.audio.transcribe(media.locator)
                    # Only index if we actually got text back (handles silent videos)
                    if transcript and transcript.strip():
                        text_embedding = model_manager.text_embed.get_embeddings(transcript)
                        self.text_collection.upsert(
                            ids=[transcript_id],
                            embeddings=[text_embedding],
                            metadatas=[{
                                "source": media.source,
                                "source_media_id": media.media_id,
                                "locator": media.locator,
                                "display_path": media.display_path,
                                "type": "audio"
                            }]
                        )
                        self.bm25_service.add_document(transcript_id, transcript)
                
                # 2. Ambient sound indexing
                ambient_id = f"{media.media_id}:::ambient"
                ambient_results = self.audio_collection.get(ids=[ambient_id])
                
                if not ambient_results["ids"]:
                    ambient_embedding = model_manager.clap.get_audio_embeddings(media.locator)
                    # Only index if the embedding is valid (not a zero-vector from a silent file)
                    if ambient_embedding and any(v != 0 for v in ambient_embedding):
                        self.audio_collection.upsert(
                            ids=[ambient_id],
                            embeddings=[ambient_embedding],
                            metadatas=[{
                                    "source": media.source,
                                    "source_media_id": media.media_id,
                                    "locator": media.locator,
                                    "display_path": media.display_path,
                                    "type": "audio"
                            }]
                        )
                
                pbar.update(1)
                
            self.bm25_service.rebuild()

    def cleanup_index(self, valid_media_items: List[MediaDescriptor]):
        """
        Removes stale entries from the index.
        """
        logger.info("Cleaning up index...")
        valid_media_ids = {m.media_id for m in valid_media_items}

        for collection in [self.image_collection, self.text_collection, self.audio_collection]:
            try:
                collection_data = collection.get()
                all_ids = collection_data["ids"]
                all_metadatas = collection_data.get("metadatas") or [None] * len(all_ids)
                
                ids_to_delete = []
                for doc_id, metadata in zip(all_ids, all_metadatas):
                    source_media_id = metadata.get("source_media_id") if metadata else doc_id
                    if source_media_id not in valid_media_ids:
                        ids_to_delete.append(doc_id)
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
            except Exception as e:
                logger.error(f"Error cleaning up collection: {e}")
            # For now, we'll just leave them in BM25 until next rebuild or implement delete in BM25Service
