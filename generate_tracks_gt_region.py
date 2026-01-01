"""
Generate trajectories STRICTLY within the Ground Truth Region (Y=899-1065).
The model runs on the full image, but we only record detections inside the target lane zone.
"""

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import cv2

def generate_tracks_gt_region(
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    model_path="visdrone_finetuned.pt",
    output_path="trajectories_exact_region_full.csv",
    conf_thresh=0.15,
    img_size=1920, 
    frames_to_process=10000 # Process ALL frames
):
    # GT Boundaries (from D2F1_lclF_v.csv analysis)
    Y_MIN = 899
    Y_MAX = 1065
    
    print(f"🚀 Tracking in GT Region Only (Y: {Y_MIN}-{Y_MAX})...")
    print(f"   Model: {model_path}")
    print(f"   Resolution: {img_size}")

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
            
            # Run Tracking (Full Frame to maintain context)
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thresh, 
                imgsz=img_size, 
                verbose=False,
                tracker="botsort.yaml"
            )[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # REGION FILTER: Only save if center is in GT Zone
                    if Y_MIN <= cy <= Y_MAX:
                        tracks.append({
                            "Frame": frame_idx,
                            "VehicleID": int(track_id),
                            "Class": int(cls),
                            "X_pixel": cx,
                            "Y_pixel": cy,
                            "Time": frame_idx / 30.0
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
        print(f"\n✅ Saved detections to {output_path}")
        print(f"   Unique IDs: {df['VehicleID'].nunique()}")
    else:
        print("\n❌ No detections found in the specified region!")

if __name__ == "__main__":
    generate_tracks_gt_region()
