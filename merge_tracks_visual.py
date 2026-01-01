"""
Refined Track Merger using Motion + Visual Appearance (Color/HOG).
Fixes fragmented tracks where one vehicle is split into multiple IDs.
"""

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def get_color_histogram(image, bins=8):
    """Calculate normalized color histogram."""
    # Convert to HSV for better color verification
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def merge_tracks(
    csv_path="generated_trajectories.csv",
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    output_path="trajectories_merged.csv",
    max_frame_gap=30,      # Max frames a car can be "lost"
    max_distance=100,      # Max pixels a car can move while lost
    visual_sim_thresh=0.7 # Minimum histogram match (0-1)
):
    print(f"🔄 Starting Track Merging...")
    print(f"   Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort by frame
    df = df.sort_values('Frame')
    unique_main_ids = df['VehicleID'].unique()
    print(f"   Initial Unique IDs: {len(unique_main_ids)}")
    
    # Pre-load video for visual features
    print("   Extracting visual features (Color Histograms)...")
    cap = cv2.VideoCapture(video_path)
    
    # specific store for last known appearance of each ID
    # ID -> Histogram
    id_appearances = {} 
    
    # Also store track "End" stats: ID -> {frame, x, y, w, h}
    track_ends = {}
    # Store track "Start" stats: ID -> {frame, x, y, w, h}
    track_starts = {}
    
    # Group by ID to find start/end points
    grouped = df.groupby('VehicleID')
    
    for vid, group in tqdm(grouped, desc="Analyzing tracks"):
        # Get start (first frame) info
        start_row = group.iloc[0]
        track_starts[vid] = {
            'frame': start_row['Frame'],
            'x': start_row['X_pixel'],
            'y': start_row['Y_pixel'],
            'row_idx': group.index[0] # To extract image later
        }
        
        # Get end (last frame) info
        end_row = group.iloc[-1]
        track_ends[vid] = {
            'frame': end_row['Frame'],
            'x': end_row['X_pixel'],
            'y': end_row['Y_pixel'],
            'row_idx': group.index[-1]
        }
        
    # We need to extract images for verification
    # Optimized: We only need images for Start of Track B and End of Track A
    # But reading random frames is slow in encoded video. 
    # Better strategy: Linear pass over video, caching only needed crops.
    
    # Identify frames we need to access
    frames_of_interest = set()
    for vid in track_ends:
        frames_of_interest.add(int(track_ends[vid]['frame']))
    for vid in track_starts:
        frames_of_interest.add(int(track_starts[vid]['frame']))
        
    sorted_frames = sorted(list(frames_of_interest))
    
    # Cache appearances
    current_frame_idx = 0
    max_interest = max(sorted_frames) if sorted_frames else 0
    
    # Store crops: key = (frame, x, y) -> histogram
    # To avoid duplicates if multiple IDs start/end same place
    spatial_hist_cache = {} 
    
    pbar = tqdm(total=max_interest)
    while True:
        ret, frame = cap.read()
        if not ret or current_frame_idx > max_interest:
            break
            
        if current_frame_idx in frames_of_interest:
            # Check ends
            # Optimization: could iterate only relevant IDs but dict lookup fast enough
            pass # We actually need to look up which IDs are here.
            
            # Let's verify start/ends at this frame
            # This is slightly inefficient O(N_cars) per frame of interest
            # Fast enough for thousands of cars
            
            # Extract histograms for tracks ending here
            for vid, data in track_ends.items():
                if data['frame'] == current_frame_idx:
                    x, y = int(data['x']), int(data['y'])
                    # Simple crop 40x40 roughly
                    h, w = 40, 40
                    y1 = max(0, y - h//2)
                    y2 = min(frame.shape[0], y + h//2)
                    x1 = max(0, x - w//2)
                    x2 = min(frame.shape[1], x + w//2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        id_appearances[f"{vid}_end"] = get_color_histogram(crop)
            
            # Extract histograms for tracks starting here
            for vid, data in track_starts.items():
                if data['frame'] == current_frame_idx:
                    x, y = int(data['x']), int(data['y'])
                    h, w = 40, 40
                    y1 = max(0, y - h//2)
                    y2 = min(frame.shape[0], y + h//2)
                    x1 = max(0, x - w//2)
                    x2 = min(frame.shape[1], x + w//2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        id_appearances[f"{vid}_start"] = get_color_histogram(crop)

        current_frame_idx += 1
        pbar.update(1)
    
    cap.release()
    
    # --- The Merge Logic ---
    
    # Map old_id -> new_id
    id_map = {vid: vid for vid in unique_main_ids}
    
    # Sort tracks by start time
    sorted_ids = sorted(unique_main_ids, key=lambda x: track_starts[x]['frame'])
    
    # Greedy matching (can use Hungarian for optimal, but greedy is fine for sequential)
    # Iterate through tracks, try to find a "parent" track that ended recently
    
    merged_count = 0
    
    active_tracks = [] # List of IDs that have ended recently
    
    for current_id in tqdm(sorted_ids, desc="Merging"):
        curr_start = track_starts[current_id]
        
        # Clean up active_tracks that are too old
        active_tracks = [pid for pid in active_tracks 
                         if (curr_start['frame'] - track_ends[pid]['frame']) <= max_frame_gap]
        
        best_match = None
        best_score = float('inf') # lower is better (distance)
        
        possible_parents = []
        
        for parent_id in active_tracks:
            parent_end = track_ends[parent_id]
            
            # Time gap
            dt = curr_start['frame'] - parent_end['frame']
            if dt <= 0: continue # Should not happen if sorted, but safety
            
            # Spatial distance
            dx = curr_start['x'] - parent_end['x']
            dy = curr_start['y'] - parent_end['y']
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Speed check (don't merge if it teleported)
            # Max speed assumption: ~20 pixels per frame?
            if dist > max_distance: continue
            
            # VISUAL CHECK
            # Compare End of Parent vs Start of Current
            hist1 = id_appearances.get(f"{parent_id}_end")
            hist2 = id_appearances.get(f"{current_id}_start")
            
            if hist1 is not None and hist2 is not None:
                sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if sim < visual_sim_thresh:
                    continue # Colors don't match
            
            possible_parents.append((parent_id, dist))
            
        # Select closest spatial match
        if possible_parents:
            possible_parents.sort(key=lambda x: x[1])
            best_match = possible_parents[0][0]
            
        if best_match:
            # Merge!
            # Current ID becomes Best Match ID
            # But wait, best_match might already be mapped to something else
            root_id = id_map[best_match]
            id_map[current_id] = root_id
            
            # Update the 'end' record of the root to be the end of current
            # so it can continue chain
            track_ends[best_match] = track_ends[current_id] 
            # (Logic caveat: we update the dict entry for the key 'best_match' 
            # to point to the new end location, effectively extending the track in our active memory)
            
            # We also need to update id_appearances if we want to chain visuals?
            # Ideally visual should be consistent, so old appearance is fine.
            # But updating to latest appearance handles lighting changes.
            id_appearances[f"{best_match}_end"] = id_appearances.get(f"{current_id}_end")
            
            merged_count += 1
        else:
            # No match, this starts a new active track
            active_tracks.append(current_id)

    print(f"✅ Merging Complete!")
    print(f"   Original IDs: {len(unique_main_ids)}")
    print(f"   Merged: {merged_count} tracks")
    # Apply new IDs
    df['VehicleID'] = df['VehicleID'].map(id_map)
    final_unique = df['VehicleID'].nunique()
    print(f"   Final Unique IDs: {final_unique}")
    
    df.to_csv(output_path, index=False)
    print(f"   Saved to {output_path}")

if __name__ == "__main__":
    merge_tracks(
        csv_path="generated_trajectories.csv",
        video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
        max_frame_gap=30,     # Allow 1 second gap (30 frames)
        max_distance=150,     # Allow large jump if speed high
        visual_sim_thresh=0.6 # Moderate visual match required
    )
