#!/usr/bin/env python3
"""
Filter tracked vehicles to match the road region in ground truth data

Analyzes the Y-coordinate range in d2f1_lclf.csv and filters tracked_500_frames.csv
to only include vehicles in the same region (top half of the road).
"""

import pandas as pd
import numpy as np

print("📊 Analyzing ground truth region...")

# Load ground truth to find the Y-coordinate range
gt = pd.read_csv('D2F1_lclF_v.csv')
print(f"Ground truth Y range: {gt['Y_pixel'].min():.1f} - {gt['Y_pixel'].max():.1f}")

# Calculate the region boundaries
y_min = gt['Y_pixel'].min()
y_max = gt['Y_pixel'].max()
y_margin = 50  # Add some margin

print(f"\n🎯 Filtering region:")
print(f"   Y_pixel range: {y_min - y_margin:.1f} to {y_max + y_margin:.1f}")

# Load tracked data
tracked = pd.read_csv('tracked_500_frames.csv')
print(f"\n📥 Original tracked data:")
print(f"   Total points: {len(tracked)}")
print(f"   Unique vehicles: {tracked['VehicleID'].nunique()}")
print(f"   Y range: {tracked['Y_pixel'].min():.1f} - {tracked['Y_pixel'].max():.1f}")

# Filter by Y coordinate
filtered = tracked[
    (tracked['Y_pixel'] >= y_min - y_margin) & 
    (tracked['Y_pixel'] <= y_max + y_margin)
].copy()

print(f"\n✂️  After filtering:")
print(f"   Total points: {len(filtered)}")
print(f"   Unique vehicles: {filtered['VehicleID'].nunique()}")
print(f"   Y range: {filtered['Y_pixel'].min():.1f} - {filtered['Y_pixel'].max():.1f}")

# Remove vehicles with too few points (likely false detections)
min_points = 5
vehicle_counts = filtered['VehicleID'].value_counts()
valid_vehicles = vehicle_counts[vehicle_counts >= min_points].index
filtered = filtered[filtered['VehicleID'].isin(valid_vehicles)]

print(f"\n🧹 After removing short tracks (< {min_points} points):")
print(f"   Total points: {len(filtered)}")
print(f"   Unique vehicles: {filtered['VehicleID'].nunique()}")

# Save filtered data
output_file = 'tracked_500_frames_filtered.csv'
filtered.to_csv(output_file, index=False)
print(f"\n✅ Saved filtered data to: {output_file}")

# Print summary by vehicle class
print("\n" + "="*60)
print("FILTERED VEHICLES BY CLASS")
print("="*60)
class_summary = filtered.groupby('Class').agg({
    'VehicleID': 'nunique',
    'Frame': 'count'
}).rename(columns={'VehicleID': 'Unique_Vehicles', 'Frame': 'Total_Points'})
print(class_summary)

# List all vehicles
print("\n" + "="*60)
print("VEHICLE DETAILS")
print("="*60)
for vid in sorted(filtered['VehicleID'].unique()):
    vid_data = filtered[filtered['VehicleID'] == vid]
    vclass = vid_data['Class'].iloc[0]
    frames = vid_data['Frame']
    print(f"{vid}: {len(vid_data)} points, frames {frames.min()}-{frames.max()}, class={vclass}")
