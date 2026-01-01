"""
Filter trajectories STRICTLY to the Ground Truth region (Y: 899-1065).
This ensures we only compare vehicles that exist in the D2F1_lclF_v.csv dataset.
"""

import pandas as pd

def filter_gt_region(
    input_csv="trajectories_lowconf_highres.csv",
    output_csv="trajectories_gt_region.csv",
    y_min=850,  # Buffer around 899
    y_max=1100  # Buffer around 1064
):
    print(f"🎯 Strict Filtering {input_csv} to GT Region (Y: {y_min}-{y_max})...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    initial_vehicles = df['VehicleID'].nunique()
    
    # Filter: Keep only points strictly inside the lane region
    filtered_df = df[
        (df['Y_pixel'] >= y_min) & 
        (df['Y_pixel'] <= y_max)
    ].copy()
    
    final_count = len(filtered_df)
    final_vehicles = filtered_df['VehicleID'].nunique()
    
    print(f"   Original: {initial_count} detections ({initial_vehicles} vehicles)")
    print(f"   Filtered: {final_count} detections ({final_vehicles} vehicles)")
    
    # Optional: Remove short tracks that just clipped the edge
    # Only keep vehicles that have meaningful presence in this zone?
    # For now, raw cut is safer.
    
    filtered_df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")

if __name__ == "__main__":
    filter_gt_region()
