from deepface import DeepFace
import os
import pickle
import numpy as np

def search_face_by_name(name_query, image_dir="face_data/", face_db_path="face_db/known_faces.pkl"):
    name = name_query.lower().strip()

    if not os.path.exists(face_db_path):
        print(f"[WARNING] Face database not found: {face_db_path}")
        return []

    with open(face_db_path, 'rb') as f:
        known_faces = pickle.load(f)

    if name not in known_faces:
        print(f"[WARNING] Face for '{name}' not found in database")
        return []

    target_embedding = np.array(known_faces[name])
    results = []

    # Search through all images in specified directories from config
    from Index.scan import read_from_csv
    
    try:
        paths, _ = read_from_csv("Index/paths.csv")
        image_paths = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except:
        # Fallback to image_dir if paths.csv not available
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in image_paths:
        try:
            # Extract face embedding from current image using DeepFace
            embeddings = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
            
            for embedding_data in embeddings:
                current_embedding = np.array(embedding_data['embedding'])
                # Calculate cosine similarity
                similarity = np.dot(target_embedding, current_embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(current_embedding))
                
                # Threshold for face match (adjust as needed)
                if similarity > 0.6:  # 60% similarity threshold
                    results.append(img_path)
                    print(f"[INFO] Face match found in {img_path} with similarity {similarity:.3f}")
                    break
        except Exception as e:
            # Skip images that can't be processed
            continue

    return results
