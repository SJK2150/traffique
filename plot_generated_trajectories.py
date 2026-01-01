"""
Plot trajectories from generated_trajectories.csv onto the video.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def plot_trajectories_on_video(
    csv_path="trajectories_lowconf_highres.csv",
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    output_path="highres_visualization.mp4",
    trail_length=30,
    show_ids=True
):
    print(f"🎬 Creating trajectory visualization...")
    print(f"   CSV: {csv_path}")
    print(f"   Video: {video_path}")
    
    # Load trajectories
    print(f"\n📊 Loading trajectories...")
    df = pd.read_csv(csv_path)
    print(f"   Total detections: {len(df)}")
    if 'VehicleID' in df.columns:
        print(f"   Unique vehicles: {df['VehicleID'].nunique()}")
    else:
        print("   'VehicleID' column missing!")
        return
        
    print(f"   Frame range: {df['Frame'].min()} - {df['Frame'].max()}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate colors
    unique_vehicles = df['VehicleID'].unique()
    np.random.seed(42)
    colors = {}
    for vehicle_id in unique_vehicles:
        colors[vehicle_id] = tuple(map(int, np.random.randint(50, 255, 3).tolist()))
    
    # Pre-process paths
    df = df.sort_values('Frame')
    vehicle_paths = defaultdict(list)
    for _, row in df.iterrows():
        vehicle_paths[row['VehicleID']].append((int(row['Frame']), int(row['X_pixel']), int(row['Y_pixel'])))
            
    frame_num = 0
    max_frame = df['Frame'].max()
    
    pbar = tqdm(total=min(total_frames, max_frame + 1), desc="Processing")
    
    while frame_num <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Current data
        frame_data = df[df['Frame'] == frame_num]
        active_ids = set(frame_data['VehicleID'].unique())
        
        # Plot trails
        for vid in active_ids:
            path = vehicle_paths[vid]
            # Get recent path points
            current_trail = []
            # Linear search backwards from end of known path for this vehicle
            # This is efficient enough for small N of vehicles per frame
            # Better: binary search or keeping track of current index, but linear is fine for this scale
            
            # Find points <= current frame and > current frame - trail_length
            # Since path is sorted by frame, we can filter easily
            valid_points = [p for p in path if frame_num - trail_length <= p[0] <= frame_num]
            
            if len(valid_points) > 1:
                color = colors[vid]
                points_coords = [(x, y) for f, x, y in valid_points]
                
                # Draw lines
                for i in range(len(points_coords)-1):
                    # Alpha calculation
                    # Point index i connects to i+1. 
                    # Let's map age match to alpha. 
                    # The latest point is at the end of the list.
                    
                    # Age of point i+1 (the newer end of segment)
                    age = len(points_coords) - 1 - (i + 1) # 0 is latest
                    
                    alpha = max(0.2, 1.0 - (age / len(points_coords)))
                    thickness = max(1, int(3 * alpha))
                    
                    cv2.line(frame, points_coords[i], points_coords[i+1], color, thickness)

        # Plot current positions
        for _, row in frame_data.iterrows():
            vehicle_id = row['VehicleID']
            x = int(row['X_pixel'])
            y = int(row['Y_pixel'])
            color = colors[vehicle_id]
            
            cv2.circle(frame, (x, y), 5, color, -1)
            
            if show_ids:
                # Text with background
                label = str(vehicle_id)
                # increased font scale 0.5 -> 0.9, thickness 1 -> 2
                font_scale = 0.9
                thickness = 2
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw black background
                cv2.rectangle(frame, (x, y-15-h), (x+w, y-15+5), (0,0,0), -1)
                
                # Draw White ID for max contrast
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_num += 1
        pbar.update(1)
        
    pbar.close()
    out.release()
    cap.release()
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    plot_trajectories_on_video(trail_length=150)
