"""
Plot Comparison with Smart GT Stitching.
If the Ground Truth is fragmented (Vehicle 1 -> Vehicle 2), this script stitches them
to show a continuous Blue line for comparison against your Generated Track.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_stitched():
    print("📊 Loading Data...")
    gt_df = pd.read_csv('D2F1_lclF_v.csv')
    gt_df.columns = gt_df.columns.str.strip()
    
    gen_df = pd.read_csv('trajectories_gt_region.csv')
    
    # We will pick 5 long generated tracks and try to find ALL matching GT segments
    print("🔍 Finding long generated tracks...")
    
    # Get top 5 longest generated tracks
    gen_counts = gen_df['VehicleID'].value_counts()
    top_gen_ids = gen_counts.head(5).index.tolist()
    
    fig, axes = plt.subplots(len(top_gen_ids), 3, figsize=(15, 4 * len(top_gen_ids)))
    fig.suptitle('Stitched Ground Truth (Blue) vs Generated (Red)', fontsize=16)
    
    for i, gen_id in enumerate(top_gen_ids):
        # Result arrays for plotting
        gt_x, gt_y, gt_t = [], [], []
        
        # Get the Generated Track
        gen_track = gen_df[gen_df['VehicleID'] == gen_id].sort_values('Frame')
        
        # FIND MATCHING GT SEGMENTS
        # We iterate through the generated track and look for ANY GT ID nearby at that frame
        matched_gt_ids = set()
        
        for _, row in gen_track.iterrows():
            frame = row['Frame']
            x, y = row['X_pixel'], row['Y_pixel']
            
            # Find GT point at this frame near this location
            # Broad search (radius 50px)
            match = gt_df[
                (gt_df['Frame'] == frame) &
                (abs(gt_df['X_pixel'] - x) < 50) &
                (abs(gt_df['Y_pixel'] - y) < 50)
            ]
            
            if not match.empty:
                # Add to plotting data
                gt_x.append(match.iloc[0]['X_pixel'])
                gt_y.append(match.iloc[0]['Y_pixel'])
                gt_t.append(match.iloc[0]['Frame'])
                matched_gt_ids.add(match.iloc[0]['VehicleID'])
        
        # Sort GT points by frame (because we appended chronologically, should be fine, but rigorous)
        if gt_t:
            sorted_indices = np.argsort(gt_t)
            gt_t = np.array(gt_t)[sorted_indices]
            gt_x = np.array(gt_x)[sorted_indices]
            gt_y = np.array(gt_y)[sorted_indices]
        
        print(f"Vehicle {i+1} (Gen ID {gen_id}): Matched GT IDs {matched_gt_ids}")
        
        # Row i, Col 0: X vs T
        ax = axes[i][0]
        if len(gt_t) > 0: ax.plot(gt_t, gt_x, 'b-', label=f'GT (Stitched)')
        ax.plot(gen_track['Frame'], gen_track['X_pixel'], 'r--', label=f'Gen ID {gen_id}')
        ax.set_title(f'Vehicle {i+1}: X vs Frame')
        ax.legend()
        
        # Row i, Col 1: Y vs T
        ax = axes[i][1]
        if len(gt_t) > 0: ax.plot(gt_t, gt_y, 'b-', label='GT')
        ax.plot(gen_track['Frame'], gen_track['Y_pixel'], 'r--', label='Gen')
        ax.set_title('Y vs Frame')
        
        # Row i, Col 2: X vs Y
        ax = axes[i][2]
        if len(gt_x) > 0: ax.plot(gt_x, gt_y, 'b-', label='GT')
        ax.plot(gen_track['X_pixel'], gen_track['Y_pixel'], 'r--', label='Gen')
        ax.set_title('X vs Y (Path)')
        ax.invert_yaxis()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('trajectory_comparison_stitched.png')
    print("✅ Saved to trajectory_comparison_stitched.png")

if __name__ == "__main__":
    plot_comparison_stitched()
