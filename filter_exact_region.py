"""
Filter trajectories to match EXACT Ground Truth boundaries.
Range: Y [899, 1065] (Based on previous analysis of D2F1_lclF_v.csv)
"""

import pandas as pd

def filter_exact():
    input_file = "trajectories_lowconf_highres.csv"
    output_file = "trajectories_first_half_exact.csv"
    
    # Exact GT Boundaries
    Y_MIN = 899
    Y_MAX = 1065
    
    print(f"✂️ Filtering {input_file} to Exact GT Region (Y: {Y_MIN}-{Y_MAX})...")
    
    df = pd.read_csv(input_file)
    
    # Filter rows based on Y position
    filtered_df = df[(df['Y_pixel'] >= Y_MIN) & (df['Y_pixel'] <= Y_MAX)]
    
    # Stats
    original_unique = df['VehicleID'].nunique()
    filtered_unique = filtered_df['VehicleID'].nunique()
    
    print(f"✅ Created {output_file}")
    print(f"   Original Vehicles (Full Video): {original_unique}")
    print(f"   Filtered Vehicles (GT Region):  {filtered_unique}")
    
    filtered_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    filter_exact()
