"""
Plot Ground Truth vs Generated Trajectories for 5 matched vehicles.
Plots X-t, Y-t, and X-Y graphs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    print("📊 Loading Data...")
    gt_df = pd.read_csv('D2F1_lclF_v.csv')
    gt_df.columns = gt_df.columns.str.strip()
    
    gen_df = pd.read_csv('trajectories_generated_exact_region.csv')
    
    # Identify matched vehicles
    # We find IDs that are spatially close in the first ~10 frames of their existence
    matches = []
    
    print("🔍 Matching vehicles...")
    
    # Iterate over top 20 GT vehicles to find matches
    gt_ids = gt_df['VehicleID'].unique()[:20]
    
    for gt_id in gt_ids:
        gt_track = gt_df[gt_df['VehicleID'] == gt_id]
        start_frame = gt_track['Frame'].min()
        start_row = gt_track.iloc[0]
        
        # Look for Generated track near this start point
        # Allow +/- 5 frames and +/- 50 pixels
        candidates = gen_df[
            (gen_df['Frame'] >= start_frame - 5) & 
            (gen_df['Frame'] <= start_frame + 5) &
            (abs(gen_df['X_pixel'] - start_row['X_pixel']) < 50) &
            (abs(gen_df['Y_pixel'] - start_row['Y_pixel']) < 50)
        ]
        
        if not candidates.empty:
            matched_gen_id = candidates.iloc[0]['VehicleID']
            matches.append((gt_id, matched_gen_id))
            if len(matches) >= 5: break
            
    print(f"✅ Found {len(matches)} matches: {matches}")
    
    # Plotting
    if not matches:
        print("❌ No matches found to plot. Check coordinate systems.")
        return

    fig, axes = plt.subplots(len(matches), 3, figsize=(15, 4 * len(matches)))
    fig.suptitle('Ground Truth (Blue) vs Generated (Red) Trajectories', fontsize=16)
    
    # Handle single row case
    if len(matches) == 1: axes = [axes]
    
    for i, (gt_id, gen_id) in enumerate(matches):
        gt_track = gt_df[gt_df['VehicleID'] == gt_id].sort_values('Frame')
        gen_track = gen_df[gen_df['VehicleID'] == gen_id].sort_values('Frame')
        
        # Row i, Col 0: X vs T
        ax = axes[i][0]
        ax.plot(gt_track['Frame'], gt_track['X_pixel'], 'b-', label=f'GT ID {gt_id}')
        ax.plot(gen_track['Frame'], gen_track['X_pixel'], 'r--', label=f'Gen ID {gen_id}')
        ax.set_title(f'Vehicle {i+1}: X vs Frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Pixel')
        ax.legend()
        
        # Row i, Col 1: Y vs T
        ax = axes[i][1]
        ax.plot(gt_track['Frame'], gt_track['Y_pixel'], 'b-', label='GT')
        ax.plot(gen_track['Frame'], gen_track['Y_pixel'], 'r--', label='Gen')
        ax.set_title(f'Vehicle {i+1}: Y vs Frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Pixel')
        
        # Row i, Col 2: X vs Y
        ax = axes[i][2]
        ax.plot(gt_track['X_pixel'], gt_track['Y_pixel'], 'b-', label='GT')
        ax.plot(gen_track['X_pixel'], gen_track['Y_pixel'], 'r--', label='Gen')
        ax.set_title(f'Vehicle {i+1}: X vs Y (Path)')
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        ax.invert_yaxis() # Image coordinates
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'trajectory_comparison_plots.png'
    plt.savefig(output_file)
    print(f"✅ Saved plots to {output_file}")

if __name__ == "__main__":
    plot_comparison()
