"""
Plot trajectories from trajectories.csv onto the video.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def plot_trajectories_on_video(
    csv_path="trajectories.csv",
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    output_path="trajectories_visualization.mp4",
    trail_length=30,
    show_ids=True
):
    """
    Plot trajectories from CSV onto video.
    """
    print(f"🎬 Creating trajectory visualization...")
    print(f"   CSV: {csv_path}")
    print(f"   Video: {video_path}")
    
    # Load trajectories
    print(f"\n📊 Loading trajectories...")
    df = pd.read_csv(csv_path)
    print(f"   Total detections: {len(df)}")
    print(f"   Unique vehicles: {df['VehicleID'].nunique()}")
    print(f"   Frame range: {df['Frame'].min()} - {df['Frame'].max()}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate colors for each vehicle
    unique_vehicles = df['VehicleID'].unique()
    np.random.seed(42)
    colors = {}
    for vehicle_id in unique_vehicles:
        colors[vehicle_id] = tuple(map(int, np.random.randint(50, 255, 3).tolist()))
    
    # Organize trajectories by frame for faster partial lookups
    # (Pre-calculating trails can be memory intensive, so we'll do per-frame lookup with efficient indexing)
    
    # Sort by frame
    df = df.sort_values('Frame')
    
    # Create a dictionary for fast access to paths
    # vehicle_id -> list of (frame, x, y)
    vehicle_paths = defaultdict(list)
    for _, row in df.iterrows():
        vehicle_paths[row['VehicleID']].append((int(row['Frame']), int(row['X_pixel']), int(row['Y_pixel'])))
            
    frame_num = 0
    max_frame = df['Frame'].max()
    
    # Since we need to plot trails, having the full path available is useful.
    # We will iterate through video frames.
    
    pbar = tqdm(total=min(total_frames, max_frame + 1), desc="Processing")
    
    while frame_num <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for current frame
        frame_data = df[df['Frame'] == frame_num]
        
        # Draw trails and active vehicles
        # To optimize, we only draw trails for vehicles active in the last `trail_length` frames
        
        # Get IDs active in current frame
        active_ids = set(frame_data['VehicleID'].unique())
        
        # Also need IDs that were active recently to show fading trails even if momentarily lost?
        # For simplicity, let's draw trails for currently visible vehicles + slightly lost ones if needed.
        # But commonly we just draw for currently visible ones to avoid clutter of "dead" tracks.
        
        for vid in active_ids:
            # Get path points up to current frame
            path = vehicle_paths[vid]
            # Filter for last N frames
            # path is list of (f, x, y), sorted by f
            
            # Find index of current frame (optimization possible)
            # Since path is sorted, we can just take the last N points that are <= frame_num
            
            # Simple linear search from end is fast enough usually
            current_trail = []
            for i in range(len(path)-1, -1, -1):
                f, x, y = path[i]
                if f > frame_num:
                    continue
                if f < frame_num - trail_length:
                    break
                current_trail.append((x, y))
            
            # current_trail is reversed (latest to oldest)
            if len(current_trail) > 1:
                color = colors[vid]
                for i in range(len(current_trail)-1):
                    pt1 = current_trail[i]
                    pt2 = current_trail[i+1]
                    # Alpha fade logic
                    # i=0 is latest (brightest), i=len is oldest (fading)
                    alpha = 1.0 - (i / len(current_trail))
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, pt1, pt2, color, thickness)

        # Draw current positions
        for _, row in frame_data.iterrows():
            vehicle_id = row['VehicleID']
            x = int(row['X_pixel'])
            y = int(row['Y_pixel'])
            color = colors[vehicle_id]
            
            # Circle
            cv2.circle(frame, (x, y), 5, color, -1)
            
            if show_ids:
                cv2.putText(frame, str(vehicle_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Info
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_num += 1
        pbar.update(1)
        
    pbar.close()
    out.release()
    cap.release()
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    plot_trajectories_on_video()
