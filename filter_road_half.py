"""
Filter trajectories to only include the top half of the road (Y < 1080).
This aligns with the region coverage of the ground truth data.
"""

import pandas as pd

def filter_top_half(
    input_csv="trajectories_lowconf_highres.csv",
    output_csv="trajectories_top_half.csv",
    y_cutoff=1080  # Midpoint of 2160p video
):
    print(f"✂️ Filtering {input_csv}...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    initial_vehicles = df['VehicleID'].nunique()
    
    # Filter condition: Keep only points in top half
    # AND re-check ground truth range (899-1064). 
    # If user wants "first half of road", usually means direction A.
    # Let's start with loose top half (Y < 1080)
    
    filtered_df = df[df['Y_pixel'] < y_cutoff].copy()
    
    final_count = len(filtered_df)
    final_vehicles = filtered_df['VehicleID'].nunique()
    
    print(f"   Original: {initial_count} detections ({initial_vehicles} vehicles)")
    print(f"   Filtered: {final_count} detections ({final_vehicles} vehicles)")
    
    filtered_df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")

if __name__ == "__main__":
    filter_top_half()
