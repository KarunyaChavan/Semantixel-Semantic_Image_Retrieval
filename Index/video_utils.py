import cv2
from PIL import Image
import os

def extract_frames_in_memory(video_path, fps=1):
    """
    Extracts frames from a video file at a specified frame rate (fps) and 
    returns them as a list of dictionaries containing the PIL Image and timestamp.
    
    Args:
        video_path (str): The path to the video file.
        fps (int, optional): The number of frames to extract per second of video. Defaults to 1.
        
    Returns:
        list: A list of dicts: {"image": PIL.Image, "timestamp": float}
              Returns an empty list if the video cannot be opened.
    """
    frames_data = []
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return frames_data
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames_data
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0 # fallback if cv2 cannot detect fps
        
    frame_interval = max(1, int(round(video_fps / fps)))
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Calculate timestamp in seconds
            timestamp_sec = frame_count / video_fps
            
            frames_data.append({
                "image": pil_image,
                "timestamp": timestamp_sec
            })
            extracted_count += 1
            
        frame_count += 1
        
    cap.release()
    return frames_data
