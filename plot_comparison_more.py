"""
Plot 5 MORE comparisons between Ground Truth and Generated Trajectories.
Skips the previously plotted vehicles to show new examples.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_more():
    print("📊 Loading Data...")
    gt_df = pd.read_csv('D2F1_lclF_v.csv')
    gt_df.columns = gt_df.columns.str.strip()
    
    gen_df = pd.read_csv('trajectories_final_full.csv')
    
    # IDs we have already seen (skip these)
    previous_gt_ids = ['Car_1', 'Car_2', 'Truck_1', 'Car_4', 'Car_9']
    
    matches = []
    
    print("🔍 Matching NEW vehicles...")
    
    # Get all unique GT IDs
    gt_ids = gt_df['VehicleID'].unique()
    
    for gt_id in gt_ids:
        # Skip if already plotted
        if gt_id in previous_gt_ids:
            continue
            
        gt_track = gt_df[gt_df['VehicleID'] == gt_id]
        if len(gt_track) < 30: continue # Skip very short GT tracks
        
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
            
            # Check if this generated ID is decent len
            gen_track_len = len(gen_df[gen_df['VehicleID'] == matched_gen_id])
            if gen_track_len > 30:
                print(f"   found match: GT {gt_id} <-> Gen {matched_gen_id}")
                matches.append((gt_id, matched_gen_id))
        
        if len(matches) >= 5: 
            break
            
    print(f"✅ Found {len(matches)} new matches.")
    
    if not matches:
        print("❌ Could not find 5 new matches.")
        return

    fig, axes = plt.subplots(len(matches), 3, figsize=(15, 4 * len(matches)))
    fig.suptitle('Ground Truth (Blue) vs Generated (Red) - 5 New Examples', fontsize=16)
    
    for i, (gt_id, gen_id) in enumerate(matches):
        gt_track = gt_df[gt_df['VehicleID'] == gt_id].sort_values('Frame')
        gen_track = gen_df[gen_df['VehicleID'] == gen_id].sort_values('Frame')
        
        # CLIP Generation to match GT range
        start_f = gt_track['Frame'].min()
        end_f = gt_track['Frame'].max()
        gen_track = gen_track[(gen_track['Frame'] >= start_f) & (gen_track['Frame'] <= end_f)]
        
        # Row i, Col 0: X vs T
        ax = axes[i][0]
        ax.plot(gt_track['Frame'], gt_track['X_pixel'], 'b-', label=f'GT {gt_id}', linewidth=2)
        ax.plot(gen_track['Frame'], gen_track['X_pixel'], 'r--', label=f'Gen {gen_id}', linewidth=2)
        ax.set_title(f'Vehicle {i+1}: X vs Frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Pixel')
        ax.legend()
        
        # Row i, Col 1: Y vs T
        ax = axes[i][1]
        ax.plot(gt_track['Frame'], gt_track['Y_pixel'], 'b-', label='GT', linewidth=2)
        ax.plot(gen_track['Frame'], gen_track['Y_pixel'], 'r--', label='Gen', linewidth=2)
        ax.set_title(f'Vehicle {i+1}: Y vs Frame')
        ax.set_xlabel('Frame')
        
        # Row i, Col 2: X vs Y
        ax = axes[i][2]
        ax.plot(gt_track['X_pixel'], gt_track['Y_pixel'], 'b-', label='GT', linewidth=2)
        ax.plot(gen_track['X_pixel'], gen_track['Y_pixel'], 'r--', label='Gen', linewidth=2)
        ax.set_title(f'Vehicle {i+1}: X vs Y (Path)')
        ax.set_xlabel('X Pixel')
        ax.invert_yaxis() # Image coordinates
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'trajectory_comparison_plots_2.png'
    plt.savefig(output_file)
    print(f"✅ Saved plots to {output_file}")

if __name__ == "__main__":
    plot_comparison_more()
