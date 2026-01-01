"""
Plot trajectories from trajectories_improved.csv onto the video.
Creates a visualization with colored trails for each vehicle.
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
    output_path="trajectories_improved_visualization.mp4",
    trail_length=30,
    show_ids=True
):
    """
    Plot trajectories from CSV onto video.
    
    Args:
        csv_path: Path to trajectories CSV
        video_path: Path to input video
        output_path: Path to output video
        trail_length: Number of frames to show in trail
        show_ids: Whether to show vehicle IDs
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
    
    # Generate colors for each vehicle (consistent across frames)
    unique_vehicles = df['VehicleID'].unique()
    np.random.seed(42)
    colors = {}
    for vehicle_id in unique_vehicles:
        colors[vehicle_id] = tuple(map(int, np.random.randint(50, 255, 3).tolist()))
    
    # Organize trajectories by frame
    print(f"\n🔄 Processing frames...")
    trajectories = defaultdict(list)  # vehicle_id -> list of (frame, x, y)
    
    for _, row in df.iterrows():
        vehicle_id = row['VehicleID']
        frame_num = int(row['Frame'])
        x = int(row['X_pixel'])
        y = int(row['Y_pixel'])
        trajectories[vehicle_id].append((frame_num, x, y))
    
    # Sort trajectories by frame
    for vehicle_id in trajectories:
        trajectories[vehicle_id].sort(key=lambda x: x[0])
    
    # Process video
    frame_num = 0
    max_frame = df['Frame'].max()
    
    pbar = tqdm(total=min(total_frames, max_frame + 1), desc="Processing")
    
    while frame_num <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for current frame
        frame_data = df[df['Frame'] == frame_num]
        
        # Draw trails for each vehicle
        for vehicle_id in trajectories:
            traj = trajectories[vehicle_id]
            
            # Get points up to current frame
            points = [(x, y) for f, x, y in traj if f <= frame_num and f >= frame_num - trail_length]
            
            if len(points) > 1:
                # Draw trail with fading effect
                color = colors[vehicle_id]
                for i in range(1, len(points)):
                    # Calculate alpha based on age
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    
                    # Draw line segment
                    cv2.line(frame, points[i-1], points[i], color, thickness)
        
        # Draw current positions and IDs
        for _, row in frame_data.iterrows():
            vehicle_id = row['VehicleID']
            x = int(row['X_pixel'])
            y = int(row['Y_pixel'])
            vehicle_class = row['Class']
            color = colors[vehicle_id]
            
            # Draw current position (larger circle)
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
            
            # Draw ID and class if enabled
            if show_ids:
                label = f"{vehicle_id}"
                # Add background for text
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x + 10, y - 20), (x + 15 + text_width, y - 20 + text_height + 5), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        info_text = f"Frame: {frame_num} | Vehicles: {len(frame_data)}"
        cv2.rectangle(frame, (10, 10), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        frame_num += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Visualization complete!")
    print(f"   Output: {output_path}")
    print(f"   Total frames processed: {frame_num}")

if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Visualization from trajectories_improved.csv")
    print("=" * 60)
    
    plot_trajectories_on_video(
        csv_path="trajectories_improved.csv",
        video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
        output_path="trajectories_improved_visualization.mp4",
        trail_length=30,  # Show last 30 frames of trail
        show_ids=True
    )
    
    print("\n" + "=" * 60)
    print("🎉 Done! You can now view the visualization.")
    print("=" * 60)
