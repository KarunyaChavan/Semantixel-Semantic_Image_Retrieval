import re
from face_recognition.face_search import search_face_by_name
from Index.create_db import create_vectordb, get_clip_text

def parse_face_query(query):
    """
    Parse a query like "Find Karunya playing cricket" to extract:
    - Person name: "Karunya"
    - Activity/context: "playing cricket"
    """
    query_lower = query.lower().strip()
    
    # Pattern to match "Find [name] [doing something]"
    pattern = r'find\s+(\w+)\s+(.+)'
    match = re.search(pattern, query_lower)
    
    if match:
        name = match.group(1)
        activity = match.group(2)
        return name, activity
    
    # Pattern to match just "Find [name]"
    pattern2 = r'find\s+(\w+)'
    match2 = re.search(pattern2, query_lower)
    
    if match2:
        name = match2.group(1)
        return name, None
    
    # If no pattern match, assume the whole query is for face search
    return query_lower.strip(), None

def integrated_face_semantic_search(query, image_collection=None, top_k=10, threshold=0.3):
    """
    Perform integrated search combining face recognition and semantic search.
    
    Args:
        query (str): Query like "Find Karunya playing cricket"
        image_collection: ChromaDB collection for semantic search
        top_k (int): Number of results to return
        threshold (float): Similarity threshold for semantic search
    
    Returns:
        list: Paths of images matching both face and semantic criteria
    """
    name, activity = parse_face_query(query)
    
    # Step 1: Find images with the specified person's face
    face_results = search_face_by_name(name)
    
    if not face_results:
        return []
    
    # If no activity specified, return face results
    if not activity:
        return face_results[:top_k]
    
    # Step 2: If activity specified, use semantic search
    if image_collection is None:
        image_collection, _ = create_vectordb("db")
    
    # Get semantic search results
    text_embedding = get_clip_text(activity)
    semantic_results = image_collection.query(text_embedding, n_results=top_k * 2)
    semantic_paths = semantic_results["ids"][0]
    semantic_similarities = [1 - d for d in semantic_results["distances"][0]]
    
    # Filter semantic results by threshold
    filtered_semantic = [
        path for path, sim in zip(semantic_paths, semantic_similarities) 
        if sim > threshold
    ]
    
    # Step 3: Find intersection of face results and semantic results  
    intersection = []
    for face_image in face_results:
        if face_image in filtered_semantic:
            intersection.append(face_image)
    
    return intersection[:top_k]

def search_by_face_and_context(name, context, image_collection=None, top_k=10, threshold=0.3):
    """
    Direct search by face name and context.
    
    Args:
        name (str): Person's name to search for
        context (str): Context/activity to search for
        image_collection: ChromaDB collection for semantic search
        top_k (int): Number of results to return
        threshold (float): Similarity threshold for semantic search
    
    Returns:
        list: Paths of images matching both criteria
    """
    query = f"find {name} {context}"
    return integrated_face_semantic_search(query, image_collection, top_k, threshold)
