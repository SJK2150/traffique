"""
Plot Single Comparison: GT Car_3 vs Gen 47.0
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_single():
    print("📊 Loading Data...")
    gt_df = pd.read_csv('D2F1_lclF_v.csv')
    gt_df.columns = gt_df.columns.str.strip()
    
    gen_df = pd.read_csv('trajectories_final_full.csv')
    
    # Target IDs
    target_gt = 'Car_6'
    target_gen = 33.0
    
    print(f"🎯 Plotting GT {target_gt} vs Gen {target_gen} (Full Duration)...")
    
    gt_track = gt_df[gt_df['VehicleID'] == target_gt].sort_values('Frame')
    gen_track = gen_df[gen_df['VehicleID'] == target_gen].sort_values('Frame')
    
    # NO CLIPPING - Show full generated track
    # start_f = gt_track['Frame'].min()
    # end_f = gt_track['Frame'].max()
    # gen_track = gen_track[(gen_track['Frame'] >= start_f) & (gen_track['Frame'] <= end_f)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Full Duration: GT {target_gt} vs Generated ID {target_gen}', fontsize=16)
    
    # X vs T
    ax = axes[0]
    ax.plot(gt_track['Frame'], gt_track['X_pixel'], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(gen_track['Frame'], gen_track['X_pixel'], 'r--', label='Generated (Full)', linewidth=2)
    ax.set_title('X vs Frame')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X Pixel')
    ax.legend()
    
    # Y vs T
    ax = axes[1]
    ax.plot(gt_track['Frame'], gt_track['Y_pixel'], 'b-', label='GT', linewidth=2)
    ax.plot(gen_track['Frame'], gen_track['Y_pixel'], 'r--', label='Gen', linewidth=2)
    ax.set_title('Y vs Frame')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y Pixel')
    
    # X vs Y
    ax = axes[2]
    ax.plot(gt_track['X_pixel'], gt_track['Y_pixel'], 'b-', label='GT', linewidth=2)
    ax.plot(gen_track['X_pixel'], gen_track['Y_pixel'], 'r--', label='Gen', linewidth=2)
    ax.set_title('X vs Y (Path)')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_file = 'full_duration_comparison_car6.png'
    plt.savefig(output_file)
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    plot_single()
