#!/usr/bin/env python3
"""
Draw tracked trajectories from CSV onto video

Reads tracked_500_d2f1_format.csv and creates a video with trajectories overlaid
"""

import pandas as pd
import cv2
import numpy as np
from collections import defaultdict
import random

# Set random seed for consistent colors
random.seed(42)

print("🎬 Creating trajectory visualization video from CSV...")

# Load tracked data
df = pd.read_csv('tracked_all_vehicles_500.csv')
print(f"✓ Loaded {len(df)} trajectory points")
print(f"✓ Unique vehicles: {df['VehicleID'].nunique()}")

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
output_path = 'trajectories_all_vehicles.mp4'
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
            
            # Keep only last 50 points for each trajectory (for cleaner visualization)
            if len(trajectory_history[vid]) > 50:
                trajectory_history[vid] = trajectory_history[vid][-50:]
    
    # Draw all trajectories
    for vid, points in trajectory_history.items():
        if len(points) < 2:
            continue
        
        color = vehicle_colors[vid]
        
        # Draw trajectory line with fading effect
        for i in range(len(points) - 1):
            # Fade older points
            alpha = (i + 1) / len(points)
            thickness = max(1, int(3 * alpha))
            
            # Blend color for fading effect
            fade_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            
            cv2.line(frame, points[i], points[i + 1], fade_color, thickness, cv2.LINE_AA)
        
        # Draw current position as a filled circle
        if points:
            current_pos = points[-1]
            # Outer glow
            cv2.circle(frame, current_pos, 8, color, -1, cv2.LINE_AA)
            # Inner white dot
            cv2.circle(frame, current_pos, 3, (255, 255, 255), -1, cv2.LINE_AA)
            # Border
            cv2.circle(frame, current_pos, 8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add info overlay
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Info panel background
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Add text
    active_vehicles = len([v for v in trajectory_history.values() if v])
    
    cv2.putText(frame, f"Frame: {frame_num}/{max_frames}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Active Vehicles: {active_vehicles}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Tracked: {len(vehicle_colors)}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)
    
    # Write frame
    out.write(frame)

# Cleanup
cap.release()
out.release()

print(f"\n\n✅ Created: {output_path}")
print(f"   Frames: {frame_num}")
print(f"   Duration: {frame_num/fps:.1f} seconds")
print(f"   Vehicles tracked: {len(vehicle_colors)}")
print(f"   Max active vehicles: {max(len([v for v in trajectory_history.values() if v]) for _ in range(1))}")

print("\n💡 Video saved! You can now play it to see all trajectories.")
print(f"   File: {output_path}")
