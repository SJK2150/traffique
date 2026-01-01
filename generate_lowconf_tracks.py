"""
Generate trajectories using the fine-tuned model with LOWER confidence threshold.
This captures the 'Red Boxes' (motors/bikes) that were previously being ignored.
"""

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import cv2

def generate_tracks_lowconf(
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    model_path="visdrone_finetuned.pt",
    output_path="trajectories_lowconf.csv",
    conf_thresh=0.15,  # LOWERED from 0.25/0.30 to catch bikes
    frames_to_process=500
):
    print(f"🚀 Starting Tracking with Low Confidence ({conf_thresh})...")
    print(f"   Model: {model_path}")
    
    model = YOLO(model_path)
    
    # Store results
    tracks = []
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Total Frames in video: {total_frames}")
    
    # Process
    pbar = tqdm(total=min(total_frames, frames_to_process))
    frame_idx = 0
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret: break
        
        # Run Tracking
        # persist=True is CRITICAL for ID continuity
        results = model.track(
            frame, 
            persist=True, 
            conf=conf_thresh, 
            verbose=False,
            tracker="botsort.yaml" # or bytetrack.yaml
        )[0]
        
        # Extract tracks
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                tracks.append({
                    "Frame": frame_idx,
                    "VehicleID": int(track_id),
                    "Class": int(cls),
                    "X_pixel": cx,
                    "Y_pixel": cy,
                    "Time": frame_idx / 30.0 # Assuming 30fps roughly
                })
                
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    
    # Save CSV
    if tracks:
        df = pd.DataFrame(tracks)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} detections to {output_path}")
        print(f"   Unique IDs: {df['VehicleID'].nunique()}")
    else:
        print("\n❌ No detections found!")

if __name__ == "__main__":
    generate_tracks_lowconf()
