#!/usr/bin/env python3
"""
Merge Fragmented Vehicle Tracks

Solves the problem where a single vehicle gets assigned different IDs at different times.
Uses spatial proximity and temporal continuity to merge fragmented tracks.

Usage:
    python merge_fragmented_tracks.py --input tracked_5_vehicles.csv --output merged_tracks.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple


class TrackMerger:
    """Merge fragmented vehicle tracks based on spatial-temporal proximity"""
    
    def __init__(self, csv_path: str, distance_threshold: float = 50, time_gap_threshold: int = 30):
        """
        Args:
            csv_path: Path to CSV with tracked vehicles
            distance_threshold: Max distance (pixels) to consider tracks as same vehicle
            time_gap_threshold: Max frame gap to bridge between track segments
        """
        self.df = pd.read_csv(csv_path)
        self.distance_threshold = distance_threshold
        self.time_gap_threshold = time_gap_threshold
        
        # Normalize column names
        self._normalize_columns()
        
        print(f"✓ Loaded {len(self.df)} tracking points")
        print(f"✓ Found {self.df['vehicle_id'].nunique()} unique vehicle IDs")
        print(f"✓ Frame range: {self.df['frame'].min()} - {self.df['frame'].max()}")
    
    def _normalize_columns(self):
        """Normalize column names to standard format"""
        rename_map = {
            'Frame': 'frame',
            'VehicleID': 'vehicle_id',
            'X_pixel': 'x_px',
            'Y_pixel': 'y_px',
            'Class': 'class',
            'Time': 'time'
        }
        self.df.rename(columns=rename_map, inplace=True)
    
    def find_mergeable_tracks(self) -> Dict[str, str]:
        """
        Find tracks that should be merged together.
        
        Returns:
            Dictionary mapping old_id -> new_id (merged ID)
        """
        print("\n🔍 Analyzing tracks for merging...")
        
        # Group by vehicle ID
        tracks = {}
        for vid, group in self.df.groupby('vehicle_id'):
            tracks[vid] = {
                'frames': sorted(group['frame'].values),
                'start_frame': group['frame'].min(),
                'end_frame': group['frame'].max(),
                'start_pos': (group[group['frame'] == group['frame'].min()]['x_px'].iloc[0],
                             group[group['frame'] == group['frame'].min()]['y_px'].iloc[0]),
                'end_pos': (group[group['frame'] == group['frame'].max()]['x_px'].iloc[0],
                           group[group['frame'] == group['frame'].max()]['y_px'].iloc[0]),
                'class': group['class'].iloc[0] if 'class' in group.columns else None,
                'data': group
            }
        
        # Find merge candidates
        merge_map = {}  # old_id -> new_id
        merged_groups = []  # List of sets of IDs that should be merged
        
        track_ids = list(tracks.keys())
        
        for i, id1 in enumerate(track_ids):
            for id2 in track_ids[i+1:]:
                if self._should_merge(tracks[id1], tracks[id2]):
                    # Find if either ID is already in a merge group
                    group1 = None
                    group2 = None
                    
                    for group in merged_groups:
                        if id1 in group:
                            group1 = group
                        if id2 in group:
                            group2 = group
                    
                    if group1 is None and group2 is None:
                        # Create new merge group
                        merged_groups.append({id1, id2})
                    elif group1 is not None and group2 is None:
                        # Add id2 to id1's group
                        group1.add(id2)
                    elif group1 is None and group2 is not None:
                        # Add id1 to id2's group
                        group2.add(id1)
                    elif group1 != group2:
                        # Merge the two groups
                        group1.update(group2)
                        merged_groups.remove(group2)
        
        # Create merge map
        for group in merged_groups:
            # Use the ID with the earliest start frame as the canonical ID
            sorted_ids = sorted(group, key=lambda x: tracks[x]['start_frame'])
            canonical_id = sorted_ids[0]
            
            for vid in group:
                merge_map[vid] = canonical_id
        
        print(f"\n✓ Found {len(merged_groups)} groups of tracks to merge")
        for group in merged_groups:
            sorted_ids = sorted(group, key=lambda x: tracks[x]['start_frame'])
            canonical = sorted_ids[0]
            print(f"  Merging {sorted_ids} -> {canonical}")
        
        return merge_map
    
    def _should_merge(self, track1: Dict, track2: Dict) -> bool:
        """
        Determine if two tracks should be merged.
        
        Criteria:
        1. Same vehicle class (if available)
        2. Temporal proximity (one ends near when the other starts)
        3. Spatial proximity (end position of one is near start position of other)
        """
        # Check class match (if available)
        if track1['class'] is not None and track2['class'] is not None:
            if track1['class'] != track2['class']:
                return False
        
        # Check temporal proximity
        # Case 1: track1 ends, then track2 starts
        time_gap_1_to_2 = track2['start_frame'] - track1['end_frame']
        # Case 2: track2 ends, then track1 starts
        time_gap_2_to_1 = track1['start_frame'] - track2['end_frame']
        
        # Check if there's temporal overlap (should NOT merge if they overlap)
        if track1['start_frame'] <= track2['end_frame'] and track2['start_frame'] <= track1['end_frame']:
            # Tracks overlap in time - same vehicle can't be in two places at once
            return False
        
        # One track should end before the other starts
        if time_gap_1_to_2 > 0 and time_gap_1_to_2 <= self.time_gap_threshold:
            # track1 ends, track2 starts - check spatial proximity
            distance = np.sqrt(
                (track1['end_pos'][0] - track2['start_pos'][0])**2 +
                (track1['end_pos'][1] - track2['start_pos'][1])**2
            )
            return distance <= self.distance_threshold
        
        elif time_gap_2_to_1 > 0 and time_gap_2_to_1 <= self.time_gap_threshold:
            # track2 ends, track1 starts - check spatial proximity
            distance = np.sqrt(
                (track2['end_pos'][0] - track1['start_pos'][0])**2 +
                (track2['end_pos'][1] - track1['start_pos'][1])**2
            )
            return distance <= self.distance_threshold
        
        return False
    
    def merge_tracks(self, merge_map: Dict[str, str]) -> pd.DataFrame:
        """
        Apply the merge map to create a new DataFrame with merged tracks.
        
        Args:
            merge_map: Dictionary mapping old_id -> new_id
        
        Returns:
            DataFrame with merged vehicle IDs
        """
        print("\n🔄 Applying track merges...")
        
        df_merged = self.df.copy()
        
        # Apply merges
        df_merged['vehicle_id'] = df_merged['vehicle_id'].apply(
            lambda x: merge_map.get(x, x)
        )
        
        # Sort by vehicle ID and frame
        df_merged = df_merged.sort_values(['vehicle_id', 'frame'])
        
        print(f"✓ Merged tracks complete")
        print(f"✓ Original IDs: {self.df['vehicle_id'].nunique()}")
        print(f"✓ After merging: {df_merged['vehicle_id'].nunique()}")
        
        return df_merged
    
    def process(self, output_path: str):
        """Complete merge process and save results"""
        merge_map = self.find_mergeable_tracks()
        
        if not merge_map:
            print("\n⚠️  No tracks to merge - all tracks are already unique")
            return self.df
        
        df_merged = self.merge_tracks(merge_map)
        
        # Save results
        df_merged.to_csv(output_path, index=False)
        print(f"\n✅ Saved merged tracks to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MERGE SUMMARY")
        print("="*60)
        for vid in sorted(df_merged['vehicle_id'].unique()):
            vid_data = df_merged[df_merged['vehicle_id'] == vid]
            print(f"{vid}: {len(vid_data)} points, frames {vid_data['frame'].min()}-{vid_data['frame'].max()}")
        
        return df_merged


def main():
    parser = argparse.ArgumentParser(description="Merge fragmented vehicle tracks")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", default="merged_tracks.csv", help="Output CSV file")
    parser.add_argument("--distance-threshold", type=float, default=50, 
                       help="Max distance (pixels) to merge tracks (default: 50)")
    parser.add_argument("--time-gap-threshold", type=int, default=30,
                       help="Max frame gap to bridge (default: 30)")
    
    args = parser.parse_args()
    
    merger = TrackMerger(
        args.input,
        distance_threshold=args.distance_threshold,
        time_gap_threshold=args.time_gap_threshold
    )
    
    merger.process(args.output)


if __name__ == "__main__":
    main()
