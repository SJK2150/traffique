import cv2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def load_and_index_csv(csv_path, label="Default"):
    print(f"Loading {label} CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Flexible column matching
    cols = {c.lower(): c for c in df.columns}
    col_map = {}
    
    # Map required columns
    if 'frame' in cols: col_map['Frame'] = cols['frame']
    elif 'Frame' in df.columns: col_map['Frame'] = 'Frame'
    
    if 'vehicleid' in cols: col_map['VehicleID'] = cols['vehicleid']
    elif 'id' in cols: col_map['VehicleID'] = cols['id']
    elif 'VehicleID' in df.columns: col_map['VehicleID'] = 'VehicleID'

    if 'x_pixel' in cols: col_map['X'] = cols['x_pixel']
    elif 'x' in cols: col_map['X'] = cols['x']
    elif 'X_pixel' in df.columns: col_map['X'] = 'X_pixel'
    
    if 'y_pixel' in cols: col_map['Y'] = cols['y_pixel']
    elif 'y' in cols: col_map['Y'] = cols['y']
    elif 'Y_pixel' in df.columns: col_map['Y'] = 'Y_pixel'
    
    # Validate
    if len(col_map) < 4:
        print(f"Error: Missing columns in {label} CSV. Mapped: {col_map}")
        return None, None

    # Indexing
    frame_data = {}
    vehicle_history = {}
    
    # Pre-process
    df[col_map['VehicleID']] = df[col_map['VehicleID']].astype(str)
    
    # Build frame index
    print(f"Indexing {label} data...")
    frames = df[col_map['Frame']].values
    vids = df[col_map['VehicleID']].values
    xs = df[col_map['X']].values
    ys = df[col_map['Y']].values
    
    for f, vid, x, y in zip(frames, vids, xs, ys):
        if f not in frame_data: frame_data[f] = []
        frame_data[f].append((vid, x, y))
        
    # Build history
    for vid, group in df.groupby(col_map['VehicleID']):
        sorted_group = group.sort_values(col_map['Frame'])
        vehicle_history[vid] = list(zip(
            sorted_group[col_map['Frame']].values,
            sorted_group[col_map['X']].values,
            sorted_group[col_map['Y']].values
        ))
        
    return frame_data, vehicle_history

def match_trajectories(data1, data2, threshold=50.0):
    """
    Find automated trajectories (data1) that match manual ones (data2).
    Returns a set of matched IDs from data1.
    """
    print("Matching trajectories...")
    matched_ids = set()
    
    # Invert data structure for faster lookup: ID -> {frame: (x, y)}
    def get_traj_map(data):
        traj_map = {}
        for frame, vehicles in data.items():
            for vid, x, y in vehicles:
                if vid not in traj_map: traj_map[vid] = {}
                traj_map[vid][frame] = (x, y)
        return traj_map

    traj1 = get_traj_map(data1)
    traj2 = get_traj_map(data2)

    for vid2, points2 in tqdm(traj2.items(), desc="Matching"):
        best_match = None
        min_dist = float('inf')
        
        # Check against all automated trajectories
        # Optimization: Could filter by time overlap first
        start_frame = min(points2.keys())
        end_frame = max(points2.keys())
        
        for vid1, points1 in traj1.items():
            # Quick overlap check
            if min(points1.keys()) > end_frame or max(points1.keys()) < start_frame:
                continue
                
            # Calculate distance on shared frames
            shared_frames = set(points2.keys()) & set(points1.keys())
            if not shared_frames: continue
            
            total_dist = 0
            count = 0
            for f in shared_frames:
                p1 = points1[f]
                p2 = points2[f]
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                total_dist += dist
                count += 1
            
            avg_dist = total_dist / count
            if avg_dist < threshold and avg_dist < min_dist:
                min_dist = avg_dist
                best_match = vid1
        
        if best_match:
            matched_ids.add(best_match)
            
    print(f"Matched {len(matched_ids)} automated trajectories out of {len(traj1)}")
    return matched_ids

def visualize_trajectories(csv_path1, video_path, output_path, limit=None, csv_path2=None, only_matches=False):
    # Load Primary CSV (Automated - Red/Orange/Yellow)
    data1, hist1 = load_and_index_csv(csv_path1, "Primary")
    if not data1: return

    # Load Comparison CSV (Manual - Green)
    data2, hist2 = (None, None)
    if csv_path2:
        data2, hist2 = load_and_index_csv(csv_path2, "Comparison")
        if not data2: return

    # Filter if requested
    valid_ids = None
    if only_matches and data2:
        valid_ids = match_trajectories(data1, data2)

    # Video Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if limit:
        total = min(total, limit)
        print(f"Limiting to first {total} frames")

    # Output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Color Config
    print("Processing video...")
    for frame_idx in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret: break
        
        # --- Draw Comparison First (Background/Green) ---
        if data2:
            vehicles = data2.get(frame_idx, [])
            for vid, cx, cy in vehicles:
                color = (0, 255, 0)
                trail = hist2.get(vid, [])
                recent = [pt for pt in trail if frame_idx - 50 <= pt[0] <= frame_idx]
                if len(recent) > 1:
                    pts = np.array([[p[1], p[2]] for p in recent], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, color, 3)
                
                center = (int(cx), int(cy))
                cv2.circle(frame, center, 6, color, -1)
                cv2.putText(frame, f"{vid}", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Draw Primary Second (Foreground/Red) ---
        vehicles = data1.get(frame_idx, [])
        for vid, cx, cy in vehicles:
            # Skip if filtering enable and not matched
            if valid_ids is not None and vid not in valid_ids:
                continue

            # Generate distinct warm colors
            np.random.seed(hash(vid) % 2**32) 
            hue = np.random.randint(0, 40)
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            
            trail = hist1.get(vid, [])
            recent = [pt for pt in trail if frame_idx - 50 <= pt[0] <= frame_idx]
            if len(recent) > 1:
                pts = np.array([[p[1], p[2]] for p in recent], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2) # Thinner line for overlay
            
            center = (int(cx), int(cy))
            cv2.circle(frame, center, 4, color, -1)
            
            # Label
            cv2.putText(frame, f"{vid}", (center[0]-20, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Legend
        if data2:
            cv2.rectangle(frame, (20, 20), (350, 100), (0, 0, 0), -1)
            cv2.putText(frame, "Comparison", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Automated (Red)", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(frame, "Manual (Green)", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Primary CSV (Automated)")
    parser.add_argument("--compare-csv", help="Comparison CSV (Manual/Truth)")
    parser.add_argument("--video", default=r"d:\traffique-traj_rmse\traffique-traj_rmse\D2F1_stab.mp4")
    parser.add_argument("--output", default=r"d:\traffique-traj_rmse\traffique-traj_rmse\output\comparison.mp4")
    parser.add_argument("--limit", type=int, help="Limit frames")
    parser.add_argument("--only-matches", action="store_true", help="Only show automated tracks that match manual ones")
    
    args = parser.parse_args()
    
    visualize_trajectories(args.csv, args.video, args.output, args.limit, args.compare_csv, args.only_matches)
