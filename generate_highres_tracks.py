"""
Generate trajectories using High Resolution Inference (imgsz=1920).
This allows the model to see small bikes without the jitter artifacts of SAHI.
"""

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import cv2

def generate_tracks_highres(
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    model_path="visdrone_finetuned.pt",
    output_path="trajectories_highres.csv",
    tracker_config="sticky_botsort.yaml",
    conf_thresh=0.15,
    img_size=1920, # <--- THE KEY CHANGE: Running at 1080p+ resolution
    frames_to_process=500
):
    print(f"🚀 Starting High-Res Tracking...")
    print(f"   Resolution: {img_size}px (standard is 640px)")
    print(f"   Model: {model_path}")
    
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
            
            # Run Tracking at High Res
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thresh, 
                imgsz=img_size, # Force high resolution
                verbose=False,
                tracker=tracker_config
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
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
    
    if tracks:
        df = pd.DataFrame(tracks)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved High-Res detections to {output_path}")
        print(f"   Unique IDs: {df['VehicleID'].nunique()}")
    else:
        print("\n❌ No detections found!")

if __name__ == "__main__":
    generate_tracks_highres()
