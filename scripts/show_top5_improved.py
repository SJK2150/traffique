#!/usr/bin/env python3
"""Show top 5 vehicles with the improved trajectories"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast

def main():
    workspace = Path(__file__).parent.parent
    csv_path = workspace / 'output' / 'D1F1_stab_cropped_improved_tracks.csv'
    video_path = workspace / 'output' / 'D1F1_stab_cropped.mp4'
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    
    print(f"📊 Loading trajectories from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Sort by trajectory length
    df_sorted = df.sort_values('trajectory_len', ascending=False)
    top5 = df_sorted.head(5)
    
    print(f"\n🎯 Top 5 vehicles by trajectory length:")
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {idx}. Vehicle {int(row['track_id'])}: "
              f"{row['trajectory_len']:.1f}px, "
              f"avg_speed: {row['avg_speed']:.2f}px/f, "
              f"frames: {row['frames']}")
    
    # Load video to get dimensions
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get middle frame for visualization
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Failed to read frame {middle_frame}")
        return
    
    print(f"\n📹 Loaded video: {width}x{height} @ {fps}fps, total {total_frames} frames")
    print(f"✅ Using frame {middle_frame} for visualization")
    
    # Draw trajectories
    vis = frame.copy()
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
    ]
    
    for idx, (_, row) in enumerate(top5.iterrows()):
        color = colors[idx % len(colors)]
        vid = int(row['track_id'])
        
        try:
            # Parse trajectory coordinates
            traj_str = row['trajectory_world']
            if isinstance(traj_str, str):
                traj = ast.literal_eval(traj_str)
            else:
                traj = traj_str
            
            if not traj or len(traj) == 0:
                continue
            
            # Draw trajectory line
            traj_pts = np.array(traj, dtype=np.int32)
            cv2.polylines(vis, [traj_pts], False, color, 2)
            
            # Draw points
            for i, pt in enumerate(traj_pts):
                if i % max(1, len(traj_pts)//5) == 0:  # Every 20%
                    cv2.circle(vis, tuple(pt), 3, color, -1)
            
            # Mark start and end
            if len(traj_pts) > 0:
                cv2.circle(vis, tuple(traj_pts[0]), 8, (0, 255, 0), -1)   # Green start
                cv2.circle(vis, tuple(traj_pts[-1]), 8, (0, 0, 255), -1)   # Red end
                
                # Label
                label = f"Car{idx+1} (ID:{vid})"
                cv2.putText(vis, label, 
                           tuple(traj_pts[0] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"⚠️  Error processing vehicle {vid}: {e}")
            continue
    
    # Save result
    output_path = workspace / 'output' / 'D1F1_stab_top5_improved_trajectories.jpg'
    cv2.imwrite(str(output_path), vis)
    print(f"\n✅ Saved visualization to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    main()
