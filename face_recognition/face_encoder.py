from deepface import DeepFace
import os
import pickle
import numpy as np

def encode_faces(face_dir="face_data/", save_path="face_db/known_faces.pkl"):
    """
    Encode faces using DeepFace with Facenet model.
    """
    face_db = {}
    
    for filename in os.listdir(face_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(face_dir, filename)
        try:
            # Extract face embedding using DeepFace
            embedding = DeepFace.represent(img_path=path, model_name='Facenet', enforce_detection=False)
            if embedding:
                # Use filename (or format like karunya.jpg → name = "karunya")
                name = os.path.splitext(filename)[0].lower()
                face_db[name] = embedding[0]['embedding']
                print(f"[INFO] Encoded face for {name}")
        except Exception as e:
            print(f"[WARNING] Could not encode {filename}: {e}")

    with open(save_path, 'wb') as f:
        pickle.dump(face_db, f)

    print(f"[INFO] Encoded and saved {len(face_db)} known faces.")

if __name__ == "__main__":
    encode_faces()
