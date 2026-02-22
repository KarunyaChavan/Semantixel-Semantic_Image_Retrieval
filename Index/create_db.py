import chromadb
from tqdm import tqdm
import os
import sys
import yaml
from Index.scan import read_from_csv
from text_embeddings.bm25_search import BM25TextIndex
import warnings

warnings.filterwarnings("ignore")

# Disable ChromaDB telemetry to avoid telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

deep_scan = config["deep_scan"]
batch_size = config["batch_size"]

if config["clip"]["provider"] == "HF_transformers":
    from CLIP.hftransformers_clip import get_clip_image, get_clip_text
elif config["clip"]["provider"] == "mobileclip":
    from CLIP.mobile_clip import get_clip_image, get_clip_text

if config["text_embed"]["provider"] == "HF_transformers":
    from text_embeddings.hftransformers_embeddings import get_text_embeddings
elif config["text_embed"]["provider"] == "ollama":
    from text_embeddings.ollama_embeddings import get_text_embeddings
elif config["text_embed"]["provider"] == "llama_cpp":
    from text_embeddings.llamacpp_embeddings import get_text_embeddings

from ocr_model.OCR import apply_OCR


def create_vectordb(path):
    """
    Create and return image and text collections in a VectorDB database.

    This function initializes a PersistentClient with the given path, and then
    gets or creates two collections: 'images' and 'texts'. Both collections
    use cosine similarity for nearest neighbor search.

    Args:
        path (str): The path to the VectorDB database.

    Returns:
        tuple: A tuple containing two Collection objects. The first element
        is the 'images' collection, and the second element is the 'texts'
        collection.
    """
    client = chromadb.PersistentClient(
        path,
    )
    image_collection = client.get_or_create_collection(
        "images", metadata={"hnsw:space": "cosine"}
    )
    text_collection = client.get_or_create_collection(
        "texts", metadata={"hnsw:space": "cosine"}
    )
    return image_collection, text_collection


def index_images(image_collection, text_collection):
    """
    Index images and extract video frames in the database.

    This function iterates over all image/video paths. For images, it processes them normally.
    For videos, it extracts frames in memory, generating a composite ID like `video.mp4:::timestamp`,
    and creates embeddings for both visual and text content. 
    Text is indexed using both semantic embeddings (Chroma) and BM25 keyword search.

    Args:
        image_collection (Collection): The image collection in the database.
        text_collection (Collection): The text collection in the database.
    """
    paths, averages = read_from_csv("paths.csv")
    bm25_index = BM25TextIndex()
    
    video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
    
    with tqdm(total=len(paths), desc="Indexing media") as pbar:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            to_process_paths = []
            
            # For collecting the actual items to feed into CLIP/OCR
            # This will be a mix of strings (image paths) and PIL Images (video frames)
            processing_inputs = []
            processing_ids = []
            processing_metadatas = []

            for path in batch_paths:
                is_video = path.lower().endswith(tuple(video_extensions))
                
                # We need a robust way to check if a video has been indexed.
                # Since we don't know the exact timestamps that were generated unless we query them,
                # we'll do a prefix query or just check if ANY frame from this video exists.
                # Since ChromaDB get() doesn't support prefix matching easily without where clauses,
                # for now, we'll check if the base path exists in metadatas or we just re-index if it's new.
                # A better approach: query by metadata where source_video = path.
                
                if is_video:
                    results = image_collection.get(where={"source_video": path})
                    if len(results["ids"]) == 0:
                        to_process_paths.append(path)
                else:
                    if len(image_collection.get(ids=[path])["ids"]) > 0:
                        if deep_scan:
                            # Check if the average pixel value has changed
                            average = image_collection.get(ids=[path])["metadatas"][0]["average"]
                            if average != averages[paths.index(path)]:
                                to_process_paths.append(path)
                    else:
                        to_process_paths.append(path)

            for path in to_process_paths:
                if path.lower().endswith(tuple(video_extensions)):
                    from Index.video_utils import extract_frames_in_memory
                    frames_data = extract_frames_in_memory(path, fps=1)
                    for frame_data in frames_data:
                        processing_inputs.append(frame_data["image"])
                        composite_id = f"{path}:::{frame_data['timestamp']}"
                        processing_ids.append(composite_id)
                        processing_metadatas.append({"average": 0, "source_video": path, "timestamp": frame_data['timestamp'], "type": "video_frame"})
                else:
                    processing_inputs.append(path)
                    processing_ids.append(path)
                    processing_metadatas.append({"average": averages[paths.index(path)], "type": "image"})

            if processing_inputs:
                # Process CLIP embeddings in batch
                image_embeddings = get_clip_image(processing_inputs)

                # Perform batch upsert for image collection
                image_collection.upsert(
                    ids=processing_ids,
                    embeddings=image_embeddings,
                    metadatas=processing_metadatas,
                )
                ocr_texts = apply_OCR(processing_inputs)

                # Process OCR and text embeddings individually
                for idx in range(len(processing_inputs)):
                    current_id = processing_ids[idx]
                    if ocr_texts[idx] is not None:
                        # Add to semantic embeddings (Chroma)
                        text_embeddings = get_text_embeddings(ocr_texts[idx])
                        text_collection.upsert(
                            ids=[current_id], embeddings=[text_embeddings],
                            metadatas=[processing_metadatas[idx]]
                        )
                        
                        # Add to BM25 index (keyword search)
                        bm25_index.add_document(current_id, ocr_texts[idx])

            pbar.update(min(batch_size, len(paths) - i))
    
    # Rebuild and save BM25 index
    bm25_index.rebuild_index()


def clean_index(image_collection, text_collection, verbose=False):
    """
    Clean up the database.

    This function iterates over all IDs in the image collection, and for each ID, it checks if the ID is in
    the list of original image paths. For video frames (IDs spanning composite format `path:::timestamp`), 
    it will check if the source video path is still valid.
    If not, it deletes the ID from the image collection and the text collection.

    Args:
        paths (list): The list of original image paths. These paths are used as IDs in the
                               image and text collections.
        image_collection (Collection): The image collection in the database.
        text_collection (Collection): The text collection in the database.
    """
    paths, averages = read_from_csv("paths.csv")
    all_image_ids = image_collection.get()["ids"]
    all_text_ids = text_collection.get()["ids"]
    
    for i, id in tqdm(
        enumerate(all_image_ids),
        total=len(all_image_ids),
        desc="Cleaning up database",
    ):
        
        # Determine the base path (for images: id, for video frames: split by ':::')
        base_path = id.split(":::")[0] if ":::" in id else id
        
        if base_path not in paths:
            # Check if ID exists in image collection before deletion
            if id in all_image_ids:
                if verbose:
                    print(f"deleting: {id} from image_collection")
                try:
                    image_collection.delete(ids=[id])
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not delete {id} from image_collection: {e}")
            
            # Check if ID exists in text collection before deletion
            if id in all_text_ids:
                if verbose:
                    print(f"deleting: {id} from text_collection")
                try:
                    text_collection.delete(ids=[id])
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not delete {id} from text_collection: {e}")
