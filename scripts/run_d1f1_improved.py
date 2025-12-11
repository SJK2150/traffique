"""
Improved D1F1 analysis with:
- Kalman-filtered tracking
- Trajectory smoothing
- Better motion prediction
"""

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import pandas as pd

from interactive_analytics import VehicleAnalyzer
from utils.reid import ReIDExtractor
from utils.improved_tracker import ImprovedOnlineTracker
from utils.trajectory import smooth_kalman, linear_interpolate, rts_smoother


def run_improved_analysis(video_name: str, frame_idx: int = 0, time_window: int = 10, 
                         confidence: float = 0.2, use_sahi: bool = True, skip_crop: bool = False):
    """Run analysis with improved tracking"""
    
    repo_root = Path(__file__).resolve().parents[1]
    uploads = repo_root / 'uploads'
    video_path = uploads / video_name
    
    if not video_path.exists():
        outdir = repo_root / 'output'
        local = Path(r'C:/Users/sakth/Documents/traffique_footage')
        if (outdir / video_name).exists():
            video_path = outdir / video_name
        elif (local / video_name).exists():
            video_path = local / video_name
        else:
            raise FileNotFoundError(f"Video not found: {video_name}")
    
    # Auto-detect if video is already cropped
    if '_cropped' in video_name:
        skip_crop = True

    print(f"Running IMPROVED analysis on: {video_path}")

    analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=use_sahi, sahi_slice_size=640)
    analyzer.load_model()

    reid = ReIDExtractor(model_path=str(repo_root / 'models' / 'osnet.pth'))
    
    # Use improved tracker with Kalman filtering
    tracker = ImprovedOnlineTracker(max_missed=int(5 * 25),  # 5 sec @ 25fps
                                   dist_thresh_px=100,        # More conservative
                                   appearance_weight=0.5)      # Balance motion and appearance

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(frame_idx)
    end_frame = int(frame_idx + max(1, int(time_window * fps)))

    print(f"Processing frames {start_frame} -> {end_frame} (fps={fps})")
    print(f"Using Kalman-filtered tracking (improved)")

    # Load constant crop if available
    constant_crop = None
    crop_json = repo_root / 'output' / 'constant_crop.json'
    if crop_json.exists() and not skip_crop:
        with open(crop_json) as f:
            constant_crop = json.load(f)

    # Detection and tracking loop
    rows = []
    total_frames = end_frame - start_frame
    
    for idx, f in enumerate(range(start_frame, end_frame + 1)):
        if idx % 25 == 0:
            pct = (idx / total_frames) * 100
            print(f'  Frame {f} ({idx}/{total_frames}, {pct:.1f}%)')
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break

        proc_frame = frame
        
        # Apply constant crop if available
        if constant_crop is not None:
            x1, y1, x2, y2 = constant_crop
            h, w = proc_frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            proc_frame = proc_frame[y1:y2, x1:x2]

        # Detect vehicles
        detections = analyzer._detect_vehicles(proc_frame)
        processed = []
        
        for d in detections:
            bbox = [int(round(x)) for x in d['bbox']]
            emb = reid.extract(proc_frame, bbox)
            processed.append({
                'bbox': tuple(bbox),
                'score': float(d.get('confidence', 0.0)),
                'class': d.get('class_name', ''),
                'embedding': emb
            })

        # Update tracker with Kalman filtering
        tracker.update(processed, frame_idx=f)

    cap.release()

    # Get all tracks
    all_tracks = tracker.get_all_tracks()
    print(f"\nTotal tracks before filtering: {len(all_tracks)}")
    
    # Post-process: smooth and filter trajectories
    filtered_tracks = []
    for track in all_tracks:
        if len(track.bboxes) >= 3:  # Require at least 3 frames
            # Smooth trajectory
            centers = np.array([((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2 in track.bboxes])
            centers_smooth = smooth_kalman(centers)
            
            filtered_tracks.append({
                'id': track.id,
                'frames': track.frames,
                'centers': centers_smooth,
                'bboxes': track.bboxes,
                'embedding': track.embedding
            })
    
    print(f"Tracks after filtering (min 3 frames): {len(filtered_tracks)}")

    # Build output rows
    for track in filtered_tracks:
        centers = track['centers']
        if isinstance(centers, np.ndarray):
            centers = centers.tolist()
        
        # Calculate average speed
        if len(centers) > 1:
            centers_arr = np.array(centers)
            diffs = np.diff(centers_arr, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            avg_speed = float(np.mean(distances))
        else:
            avg_speed = 0.0
        
        rows.append({
            'track_id': track['id'],
            'frames': json.dumps(track['frames']),
            'avg_speed': avg_speed,
            'trajectory_len': len(centers),
            'trajectory_world': json.dumps(centers)
        })

    # Save CSV
    out_dir = repo_root / 'output'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    video_name_clean = Path(video_name).stem
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{video_name_clean}_improved_tracks.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved improved CSV: {csv_path}")

    # Create visualization
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, vis_frame = cap.read()
    cap.release()
    
    if not ret:
        vis_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        if constant_crop is not None:
            x1, y1, x2, y2 = constant_crop
            h, w = vis_frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            vis_frame = vis_frame[y1:y2, x1:x2]

    # Draw trajectories with improved quality
    colors = {}
    for track in filtered_tracks:
        colors[track['id']] = (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255)
        )

    for track in filtered_tracks:
        cid = track['id']
        centers_px = track['centers']
        col = colors[cid]

        if len(centers_px) > 0:
            pts = [(int(x), int(y)) for x, y in centers_px]

            for i in range(1, len(pts)):
                cv2.line(vis_frame, pts[i - 1], pts[i], col, 2)
            if pts:
                cv2.circle(vis_frame, pts[-1], 6, col, -1)
                cv2.putText(vis_frame, f"#{cid}", (pts[-1][0] + 6, pts[-1][1] - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Save visualization
    vis_path = out_dir / f"{video_name_clean}_improved_trajectories.jpg"
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"Saved visualization: {vis_path}")
    
    return csv_path, vis_path, filtered_tracks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='D1F1_stab_cropped.mp4')
    parser.add_argument('--frame', type=int, default=9861)
    parser.add_argument('--time_window', type=int, default=15)
    parser.add_argument('--confidence', type=float, default=0.2)
    parser.add_argument('--use_sahi', action='store_true')
    parser.add_argument('--skip_crop', action='store_true')

    args = parser.parse_args()

    csv_path, vis_path, results = run_improved_analysis(
        args.video, 
        frame_idx=args.frame, 
        time_window=args.time_window, 
        confidence=args.confidence, 
        use_sahi=args.use_sahi, 
        skip_crop=args.skip_crop
    )
    print(f'Done. Results: {len(results)} tracks')
