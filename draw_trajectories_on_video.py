#!/usr/bin/env python3
"""
Draw tracked trajectories on video frames

Creates an output video with all vehicle trajectories drawn as colored lines
"""

import pandas as pd
import cv2
import numpy as np
from collections import defaultdict
import random

# Set random seed for consistent colors
random.seed(42)

print("🎬 Creating trajectory visualization video...")

# Load tracked data
df = pd.read_csv('tracked_road_region.csv')
print(f"✓ Loaded {df['VehicleID'].nunique()} vehicles")

# Assign colors to each vehicle
vehicle_colors = {}
for vid in df['VehicleID'].unique():
    # Generate bright, distinct colors
    vehicle_colors[vid] = (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

# Organize data by frame
frame_data = defaultdict(list)
for _, row in df.iterrows():
    frame_data[int(row['Frame'])].append({
        'vid': row['VehicleID'],
        'x': row['X_pixel'],
        'y': row['Y_pixel'],
        'class': row['Class']
    })

# Build trajectory history for each vehicle
trajectory_history = defaultdict(list)

# Open video
video_path = r"C:\Users\sakth\Documents\traffique_footage\D2F1_stab.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ Video: {width}x{height} @ {fps} FPS")

# Create output video writer
output_path = 'tracked_trajectories_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"\n🎥 Processing frames...")

frame_num = 0
max_frames = 500

while frame_num < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    
    # Progress indicator
    if frame_num % 50 == 0:
        print(f"  Frame {frame_num}/{max_frames}...", end='\r')
    
    # Update trajectory history for vehicles in this frame
    if frame_num in frame_data:
        for det in frame_data[frame_num]:
            vid = det['vid']
            trajectory_history[vid].append((int(det['x']), int(det['y'])))
            
            # Keep only last 30 points for each trajectory (for cleaner visualization)
            if len(trajectory_history[vid]) > 30:
                trajectory_history[vid] = trajectory_history[vid][-30:]
    
    # Draw all trajectories
    for vid, points in trajectory_history.items():
        if len(points) < 2:
            continue
        
        color = vehicle_colors[vid]
        
        # Draw trajectory line
        for i in range(len(points) - 1):
            # Fade older points
            alpha = (i + 1) / len(points)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(frame, points[i], points[i + 1], color, thickness, cv2.LINE_AA)
        
        # Draw current position as a circle
        if points:
            cv2.circle(frame, points[-1], 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, points[-1], 6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add frame number and vehicle count
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Vehicles: {len([v for v in trajectory_history.values() if v])}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write frame
    out.write(frame)

# Cleanup
cap.release()
out.release()

print(f"\n\n✅ Created: {output_path}")
print(f"   Frames: {frame_num}")
print(f"   Duration: {frame_num/fps:.1f} seconds")
print(f"   Vehicles tracked: {len(vehicle_colors)}")

print("\n💡 You can now play the video to see all trajectories!")
