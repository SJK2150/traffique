#!/usr/bin/env python3
"""
Filter tracked data to match ground truth region and vehicle types
"""

import pandas as pd

print("🎯 Filtering tracked data to match ground truth...\n")

# Load data
gt = pd.read_csv('D2F1_lclF_v.csv')
tracked = pd.read_csv('tracked_500_frames.csv')

# Get ground truth constraints
gt_500 = gt[gt['Frame'] <= 500]
y_min = gt_500['Y_pixel'].min()
y_max = gt_500['Y_pixel'].max()

print(f"Ground truth region:")
print(f"  Y range: {y_min:.1f} - {y_max:.1f}")
print(f"  Vehicle types: {sorted(gt_500['Class'].unique())}")
print(f"  Total vehicles: {gt_500['VehicleID'].nunique()}")

# Filter tracked data
# 1. Filter by Y coordinate (road region)
y_margin = 100  # Add margin to catch vehicles near boundaries
filtered = tracked[
    (tracked['Y_pixel'] >= y_min - y_margin) & 
    (tracked['Y_pixel'] <= y_max + y_margin)
].copy()

print(f"\n✂️  After Y-coordinate filter ({y_min-y_margin:.1f} - {y_max+y_margin:.1f}):")
print(f"  Vehicles: {filtered['VehicleID'].nunique()}")

# 2. Filter by vehicle type (exclude pedestrians, bicycles, tricycles)
vehicle_types = ['Car', 'Van', 'Truck', 'Bus', 'Motor']
filtered = filtered[filtered['Class'].isin(vehicle_types)]

print(f"\n🚗 After vehicle type filter (only {vehicle_types}):")
print(f"  Vehicles: {filtered['VehicleID'].nunique()}")

# 3. Remove very short tracks (likely noise)
min_points = 10
vehicle_counts = filtered['VehicleID'].value_counts()
valid_vehicles = vehicle_counts[vehicle_counts >= min_points].index
filtered = filtered[filtered['VehicleID'].isin(valid_vehicles)]

print(f"\n🧹 After removing short tracks (< {min_points} points):")
print(f"  Vehicles: {filtered['VehicleID'].nunique()}")
print(f"  Total points: {len(filtered)}")

# 4. Filter by confidence
if 'Confidence' in filtered.columns:
    min_conf = 0.3
    filtered = filtered[filtered['Confidence'] >= min_conf]
    print(f"\n📊 After confidence filter (>= {min_conf}):")
    print(f"  Vehicles: {filtered['VehicleID'].nunique()}")

# Save filtered data
output_file = 'tracked_500_frames_filtered.csv'
filtered.to_csv(output_file, index=False)

print(f"\n✅ Saved to: {output_file}")

# Summary by class
print("\n" + "="*60)
print("FILTERED VEHICLES BY CLASS")
print("="*60)
for cls in sorted(filtered['Class'].unique()):
    cls_data = filtered[filtered['Class'] == cls]
    print(f"{cls}: {cls_data['VehicleID'].nunique()} vehicles")

# Compare with ground truth
print("\n" + "="*60)
print("COMPARISON WITH GROUND TRUTH")
print("="*60)
print(f"Ground truth vehicles (first 500 frames): {gt_500['VehicleID'].nunique()}")
print(f"Tracked vehicles (filtered): {filtered['VehicleID'].nunique()}")
print(f"Difference: {gt_500['VehicleID'].nunique() - filtered['VehicleID'].nunique()}")

print("\n💡 Next steps:")
print("1. Run merge_fragmented_tracks.py to merge split tracks")
print("2. This should get closer to the 78 vehicles in ground truth")
