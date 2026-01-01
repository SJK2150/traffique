"""
Generate trajectories using 'Sticky' Tracker settings.
This solves the root cause of duplication by making the tracker 'remember' lost objects longer.
"""

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import cv2

def generate_tracks_sticky(
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    model_path="visdrone_finetuned.pt",
    output_path="trajectories_sticky.csv",
    tracker_config="sticky_botsort.yaml",
    conf_thresh=0.15, 
    frames_to_process=500
):
    print(f"🚀 Starting Sticky Tracking...")
    print(f"   Model: {model_path}")
    print(f"   Tracker: {tracker_config}")
    
    model = YOLO(model_path)
    
    tracks = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = tqdm(total=min(total_frames, frames_to_process))
    frame_idx = 0
    
    try:
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret: break
            
            # Run Tracking with CUSTOM Configuration
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thresh, 
                verbose=False,
                tracker=tracker_config  # Load our custom sticky settings
            )[0]
            
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
                        "Y_pixel": cy
                    })
                    
            frame_idx += 1
            pbar.update(1)
            
    except Exception as e:
        print(f"\n❌ Error during tracking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
    
    if tracks:
        df = pd.DataFrame(tracks)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved detections to {output_path}")
        print(f"   Unique IDs: {df['VehicleID'].nunique()}")
        
        # Compare with previous run to show improvement
        # (Assuming previous had more IDs due to fragmentation)
        print(f"   (If this number is lower than before, we successfully fixed duplicates!)")
    else:
        print("\n❌ No detections found!")

if __name__ == "__main__":
    generate_tracks_sticky()
