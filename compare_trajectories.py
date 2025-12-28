#!/usr/bin/env python3
"""
Compare Two Trajectory CSVs with Spatial Vehicle Matching (OPTIMIZED)

Matches vehicles between two tracking results based on spatial overlap,
then computes RMSE and other accuracy metrics.

Usage:
  python compare_trajectories.py --csv1 my_tracking.csv --csv2 friend_tracking.csv --output output/comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import seaborn as sns

sns.set_style("whitegrid")


class TrajectoryComparator:
    """Compare two trajectory CSVs with automatic vehicle matching"""
    
    def __init__(self, csv1_path: str, csv2_path: str, format1: str = "auto", format2: str = "auto"):
        self.df1 = pd.read_csv(csv1_path)
        self.df2 = pd.read_csv(csv2_path)
        
        # Auto-detect format
        self.format1 = self._detect_format(self.df1) if format1 == "auto" else format1
        self.format2 = self._detect_format(self.df2) if format2 == "auto" else format2
        
        # Normalize to common column names
        self.df1 = self._normalize_columns(self.df1, self.format1, label="CSV1")
        self.df2 = self._normalize_columns(self.df2, self.format2, label="CSV2")
        
        print(f"✓ Loaded CSV1: {len(self.df1)} points ({self.format1} format)")
        print(f"✓ Loaded CSV2: {len(self.df2)} points ({self.format2} format)")
        print(f"✓ Frame range CSV1: {self.df1['frame'].min()} - {self.df1['frame'].max()}")
        print(f"✓ Frame range CSV2: {self.df2['frame'].min()} - {self.df2['frame'].max()}")
    
    def _detect_format(self, df: pd.DataFrame) -> str:
        if 'X_pixel' in df.columns and 'Y_pixel' in df.columns:
            return "d2f1"
        elif 'x_px' in df.columns and 'y_px' in df.columns:
            return "default"
        else:
            raise ValueError(f"Unknown CSV format. Columns: {list(df.columns)}")
    
    def _normalize_columns(self, df: pd.DataFrame, format_type: str, label: str) -> pd.DataFrame:
        df = df.copy()
        if format_type == "d2f1":
            df.rename(columns={
                'Frame': 'frame',
                'VehicleID': 'vehicle_id',
                'X_pixel': 'x_px',
                'Y_pixel': 'y_px',
                'Class': 'class'
            }, inplace=True)
        
        df['source'] = label
        return df
    
    def match_vehicles_spatially(self, iou_threshold: float = 0.3, min_overlap_frames: int = 5) -> Dict:
        """
        OPTIMIZED: Match vehicles using frame indexing to avoid O(N*M) scans.
        """
        print(f"\n🔗 Matching vehicles between CSVs (Optimized)...")
        print(f"   Parameters: IoU threshold={iou_threshold}, min overlap frames={min_overlap_frames}")

        # 1. Pre-group data by Vehicle ID (Fast lookup)
        print("   -> Indexing vehicle trajectories...")
        grouped_1 = dict(tuple(self.df1.groupby('vehicle_id')))
        grouped_2 = dict(tuple(self.df2.groupby('vehicle_id')))

        # 2. Build a Frame Index (Frame -> List of Vehicle IDs)
        # This tells us exactly which cars are present in frame 100, so we don't check all 5000 cars.
        print("   -> Building spatial index...")
        frame_index_2 = {}
        for vid, df in grouped_2.items():
            start_f = df['frame'].min()
            end_f = df['frame'].max()
            # We store the ID in buckets of 100 frames to save memory/time
            for f_bucket in range(start_f // 100, (end_f // 100) + 1):
                if f_bucket not in frame_index_2:
                    frame_index_2[f_bucket] = set()
                frame_index_2[f_bucket].add(vid)

        matches = {}
        match_scores = {}
        
        # 3. Match
        total_v1 = len(grouped_1)
        print(f"   -> Processing {total_v1} vehicles from CSV1...")
        
        for idx, (v1, traj1) in enumerate(grouped_1.items()):
            if idx % 50 == 0: print(f"      Progress: {idx}/{total_v1}", end='\r')
            
            # Get candidate matches (only cars that exist in the same time buckets)
            start_bucket = traj1['frame'].min() // 100
            end_bucket = traj1['frame'].max() // 100
            candidates = set()
            for b in range(start_bucket, end_bucket + 1):
                if b in frame_index_2:
                    candidates.update(frame_index_2[b])
            
            if not candidates:
                continue

            # Convert traj1 to a dict for O(1) coordinate lookup: {frame: (x, y)}
            traj1_dict = dict(zip(traj1['frame'], zip(traj1['x_px'], traj1['y_px'])))
            traj1_frames = set(traj1_dict.keys())

            best_match = None
            best_score = 0.0
            
            for v2 in candidates:
                traj2 = grouped_2[v2]
                traj2_dict = dict(zip(traj2['frame'], zip(traj2['x_px'], traj2['y_px'])))
                traj2_frames = set(traj2_dict.keys())

                # Quick check: Frame overlap
                common_frames = traj1_frames.intersection(traj2_frames)
                if len(common_frames) < min_overlap_frames:
                    continue
                
                # Compute distance only on common frames
                distances = []
                for f in common_frames:
                    p1 = np.array(traj1_dict[f])
                    p2 = np.array(traj2_dict[f])
                    distances.append(np.linalg.norm(p1 - p2))
                
                avg_dist = np.mean(distances)
                
                # Score calculation
                score = 1.0 / (1.0 + avg_dist / 100.0)
                score *= min(1.0, len(common_frames) / 50.0)
                
                if score > best_score and score > iou_threshold:
                    best_score = score
                    best_match = v2
            
            if best_match is not None:
                matches[v1] = best_match
                match_scores[v1] = best_score

        print(f"\n   ✅ Matched {len(matches)} vehicle pairs")
        return matches
    
    def compute_rmse_for_matches(self, matches: Dict) -> Dict:
        """Compute RMSE for each matched vehicle pair"""
        print(f"\n📐 Computing RMSE for matched pairs...")
        
        results = {}
        
        # Pre-group df2 for fast access
        grouped_2 = dict(tuple(self.df2.groupby('vehicle_id')))
        
        for v1, v2 in matches.items():
            traj1 = self.df1[self.df1['vehicle_id'] == v1].sort_values('frame')
            traj2 = grouped_2[v2].sort_values('frame')
            
            # Find common frames
            frames1 = set(traj1['frame'].values)
            frames2 = set(traj2['frame'].values)
            common_frames = frames1.intersection(frames2)
            
            if len(common_frames) == 0:
                continue
            
            traj1_common = traj1[traj1['frame'].isin(common_frames)].sort_values('frame')
            traj2_common = traj2[traj2['frame'].isin(common_frames)].sort_values('frame')
            
            merged = pd.merge(
                traj1_common[['frame', 'x_px', 'y_px']],
                traj2_common[['frame', 'x_px', 'y_px']],
                on='frame',
                suffixes=('_1', '_2')
            )
            
            if len(merged) == 0:
                continue
            
            errors = np.sqrt(
                (merged['x_px_1'] - merged['x_px_2'])**2 +
                (merged['y_px_1'] - merged['y_px_2'])**2
            )
            
            results[f"{v1}->{v2}"] = {
                'v1': v1,
                'v2': v2,
                'n_points': len(merged),
                'rmse': np.sqrt(np.mean(errors**2)),
                'mae': np.mean(errors),
                'errors': errors.values
            }
        
        if results:
            all_rmse = [r['rmse'] for r in results.values()]
            print(f"\n   Overall RMSE statistics:")
            print(f"      Mean RMSE: {np.mean(all_rmse):.2f} px")
            print(f"      Median RMSE: {np.median(all_rmse):.2f} px")
            print(f"      Std RMSE: {np.std(all_rmse):.2f} px")
        
        return results
    
    def plot_comparison(self, matches: Dict, rmse_results: Dict, output_dir: str = "output/comparison"):
        """Generate comparison visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not rmse_results:
            print("No matched pairs to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        all_rmse = [r['rmse'] for r in rmse_results.values()]
        all_errors = np.concatenate([r['errors'] for r in rmse_results.values()])
        
        # 1. RMSE per vehicle pair
        axes[0, 0].bar(range(len(all_rmse)), sorted(all_rmse, reverse=True))
        axes[0, 0].axhline(np.mean(all_rmse), color='red', linestyle='--', label=f'Mean: {np.mean(all_rmse):.1f}px')
        axes[0, 0].set_xlabel('Vehicle Pair (sorted)')
        axes[0, 0].set_ylabel('RMSE (pixels)')
        axes[0, 0].set_title(f'RMSE per Matched Vehicle Pair (n={len(all_rmse)})')
        axes[0, 0].legend()
        
        # 2. Error distribution
        axes[0, 1].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Position Error (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Overall Error Distribution')
        
        # 3. Cumulative error
        sorted_errors = np.sort(all_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, linewidth=2)
        axes[1, 0].axhline(0.95, color='orange', linestyle='--', label='95%')
        axes[1, 0].set_xlabel('Position Error (pixels)')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Error Distribution')
        axes[1, 0].legend()
        
        # 4. RMSE vs trajectory length
        lengths = [r['n_points'] for r in rmse_results.values()]
        rmses = [r['rmse'] for r in rmse_results.values()]
        axes[1, 1].scatter(lengths, rmses, alpha=0.6)
        axes[1, 1].set_xlabel('Trajectory Length (points)')
        axes[1, 1].set_ylabel('RMSE (pixels)')
        axes[1, 1].set_title('RMSE vs Trajectory Length')
        
        plt.tight_layout()
        plt.savefig(output_path / 'trajectory_comparison.png', dpi=100)
        plt.close()
        print(f"\n   📊 Saved: trajectory_comparison.png")

    def generate_report(self, output_dir: str):
        matches = self.match_vehicles_spatially()
        if len(matches) == 0:
            print("No matches found.")
            return
        rmse_results = self.compute_rmse_for_matches(matches)
        self.plot_comparison(matches, rmse_results, output_dir)
        print(f"\n✅ Comparison complete! Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare two trajectory CSVs")
    parser.add_argument("--csv1", required=True)
    parser.add_argument("--csv2", required=True)
    parser.add_argument("--output", default="output/comparison")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-overlap", type=int, default=5)
    
    args = parser.parse_args()
    
    comparator = TrajectoryComparator(args.csv1, args.csv2)
    comparator.generate_report(args.output)

if __name__ == "__main__":
    main()