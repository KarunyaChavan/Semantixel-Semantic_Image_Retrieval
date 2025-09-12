from Index.create_db import *
from Index.scan import scan_and_save

import time
import warnings
import logging
import os

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress specific ChromaDB logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.CRITICAL)

scanned = scan_and_save()
if not scanned:
    raise Exception("Error scanning images")
image_collection, text_collection = create_vectordb("db")
start = time.time()
index_images(image_collection, text_collection)
clean_index(image_collection, text_collection)
end = time.time()
print(f"Indexing took {end - start} seconds")
