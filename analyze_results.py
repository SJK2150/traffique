"""
Utility script to analyze traffic data CSV and generate reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime


class TrafficDataAnalyzer:
    """
    Analyzer for traffic data CSV output.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize analyzer.
        
        Args:
            csv_path: Path to traffic_data.csv
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        print(f"Loaded data: {len(self.df)} records")
        print(f"Unique vehicles: {self.df['vehicle_id'].nunique()}")
        print(f"Frame range: {self.df['frame'].min()} - {self.df['frame'].max()}")
    
    def generate_summary_statistics(self) -> dict:
        """Generate summary statistics."""
        stats = {
            'total_records': len(self.df),
            'unique_vehicles': self.df['vehicle_id'].nunique(),
            'frame_range': (self.df['frame'].min(), self.df['frame'].max()),
            'by_class': self.df.groupby('class')['vehicle_id'].nunique().to_dict(),
            'avg_trajectory_length': self.df.groupby('vehicle_id')['trajectory_length'].max().mean(),
            'total_distance': self.df.groupby('vehicle_id')['total_distance'].max().sum(),
            'avg_speed': self.df['average_speed'].mean()
        }
        
        return stats
    
    def print_summary(self):
        """Print summary statistics."""
        stats = self.generate_summary_statistics()
        
        print("\n" + "="*60)
        print("Traffic Data Summary")
        print("="*60)
        print(f"\nTotal Records: {stats['total_records']}")
        print(f"Unique Vehicles: {stats['unique_vehicles']}")
        print(f"Frame Range: {stats['frame_range'][0]} - {stats['frame_range'][1]}")
        
        print("\nVehicles by Class:")
        for cls, count in stats['by_class'].items():
            print(f"  {cls}: {count}")
        
        print(f"\nAverage Trajectory Length: {stats['avg_trajectory_length']:.1f} frames")
        print(f"Total Distance Traveled: {stats['total_distance']:.1f} meters")
        print(f"Average Speed: {stats['avg_speed']:.2f} m/frame")
        print("="*60 + "\n")
    
    def plot_vehicle_counts_over_time(self, output_path: str = None):
        """Plot vehicle counts over time."""
        plt.figure(figsize=(12, 6))
        
        # Count vehicles per frame
        counts = self.df.groupby('frame')['vehicle_id'].nunique()
        
        plt.plot(counts.index, counts.values, linewidth=2)
        plt.xlabel('Frame')
        plt.ylabel('Number of Vehicles')
        plt.title('Vehicle Count Over Time')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, output_path: str = None):
        """Plot distribution of vehicle classes."""
        plt.figure(figsize=(10, 6))
        
        class_counts = self.df.groupby('class')['vehicle_id'].nunique()
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        plt.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
        plt.xlabel('Vehicle Class')
        plt.ylabel('Count')
        plt.title('Distribution of Vehicle Classes')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (cls, count) in enumerate(class_counts.items()):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_trajectory_lengths(self, output_path: str = None):
        """Plot distribution of trajectory lengths."""
        plt.figure(figsize=(10, 6))
        
        max_lengths = self.df.groupby('vehicle_id')['trajectory_length'].max()
        
        plt.hist(max_lengths, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        plt.xlabel('Trajectory Length (frames)')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Trajectory Lengths')
        plt.axvline(max_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {max_lengths.mean():.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_speed_distribution(self, output_path: str = None):
        """Plot distribution of vehicle speeds."""
        plt.figure(figsize=(10, 6))
        
        # Get final speed for each vehicle
        speeds = self.df.groupby('vehicle_id')['average_speed'].last()
        speeds = speeds[speeds > 0]  # Remove stationary
        
        plt.hist(speeds, bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
        plt.xlabel('Average Speed (m/frame)')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Vehicle Speeds')
        plt.axvline(speeds.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {speeds.mean():.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_camera_coverage(self, output_path: str = None):
        """Plot distribution across cameras."""
        plt.figure(figsize=(10, 6))
        
        camera_data = self.df.groupby(['camera_id', 'class'])['vehicle_id'].nunique().unstack(fill_value=0)
        
        camera_data.plot(kind='bar', stacked=True, alpha=0.8, 
                        color=['#2ecc71', '#3498db', '#e74c3c'])
        plt.xlabel('Camera ID')
        plt.ylabel('Number of Vehicles')
        plt.title('Vehicle Distribution Across Cameras')
        plt.legend(title='Class')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_spatial_heatmap(self, output_path: str = None, bins: int = 50):
        """Plot spatial heatmap of vehicle positions."""
        plt.figure(figsize=(12, 10))
        
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            self.df['global_x'], 
            self.df['global_y'],
            bins=bins
        )
        
        # Plot
        plt.imshow(h.T, origin='lower', cmap='hot', interpolation='bilinear',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(label='Frequency')
        plt.xlabel('Global X')
        plt.ylabel('Global Y')
        plt.title('Spatial Distribution Heatmap')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_vehicle_summaries(self, output_path: str):
        """Export per-vehicle summary."""
        vehicle_summary = self.df.groupby('vehicle_id').agg({
            'class': 'first',
            'trajectory_length': 'max',
            'total_distance': 'max',
            'average_speed': 'last',
            'frame': ['min', 'max'],
            'camera_id': lambda x: x.nunique()
        }).reset_index()
        
        vehicle_summary.columns = [
            'vehicle_id', 'class', 'trajectory_length', 
            'total_distance', 'average_speed',
            'first_frame', 'last_frame', 'cameras_visited'
        ]
        
        vehicle_summary['duration'] = vehicle_summary['last_frame'] - vehicle_summary['first_frame']
        
        vehicle_summary.to_csv(output_path, index=False)
        print(f"✓ Vehicle summary exported: {output_path}")
    
    def generate_report(self, output_dir: str):
        """Generate complete analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating analysis report...")
        
        # Summary statistics
        self.print_summary()
        
        # Generate plots
        self.plot_vehicle_counts_over_time(output_path / "vehicle_counts_over_time.png")
        self.plot_class_distribution(output_path / "class_distribution.png")
        self.plot_trajectory_lengths(output_path / "trajectory_lengths.png")
        self.plot_speed_distribution(output_path / "speed_distribution.png")
        self.plot_camera_coverage(output_path / "camera_coverage.png")
        self.plot_spatial_heatmap(output_path / "spatial_heatmap.png")
        
        # Export summaries
        self.export_vehicle_summaries(output_path / "vehicle_summaries.csv")
        
        # Create HTML report
        self.create_html_report(output_path / "report.html", output_path)
        
        print(f"\n✓ Report generated: {output_dir}")
    
    def create_html_report(self, output_path: str, image_dir: Path):
        """Create HTML report with all visualizations."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .stat-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .stat-value {{
            color: #2c3e50;
        }}
        .plot {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #95a5a6;
            text-align: right;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>🚗 Traffic Analysis Report</h1>
    
    <div class="stats">
        <h2>Summary Statistics</h2>
"""
        
        stats = self.generate_summary_statistics()
        
        html_content += f"""
        <div class="stat-row">
            <span class="stat-label">Total Records:</span>
            <span class="stat-value">{stats['total_records']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Unique Vehicles:</span>
            <span class="stat-value">{stats['unique_vehicles']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Cars:</span>
            <span class="stat-value">{stats['by_class'].get('Car', 0)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Bikes:</span>
            <span class="stat-value">{stats['by_class'].get('Bike', 0)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Pedestrians:</span>
            <span class="stat-value">{stats['by_class'].get('Pedestrian', 0)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Trajectory Length:</span>
            <span class="stat-value">{stats['avg_trajectory_length']:.1f} frames</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Distance Traveled:</span>
            <span class="stat-value">{stats['total_distance']:.1f} meters</span>
        </div>
    </div>
    
    <h2>Visualizations</h2>
    
    <div class="plot">
        <h3>Vehicle Count Over Time</h3>
        <img src="vehicle_counts_over_time.png" alt="Vehicle Count Over Time">
    </div>
    
    <div class="plot">
        <h3>Vehicle Class Distribution</h3>
        <img src="class_distribution.png" alt="Class Distribution">
    </div>
    
    <div class="plot">
        <h3>Trajectory Length Distribution</h3>
        <img src="trajectory_lengths.png" alt="Trajectory Lengths">
    </div>
    
    <div class="plot">
        <h3>Speed Distribution</h3>
        <img src="speed_distribution.png" alt="Speed Distribution">
    </div>
    
    <div class="plot">
        <h3>Camera Coverage</h3>
        <img src="camera_coverage.png" alt="Camera Coverage">
    </div>
    
    <div class="plot">
        <h3>Spatial Heatmap</h3>
        <img src="spatial_heatmap.png" alt="Spatial Heatmap">
    </div>
    
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze traffic data CSV and generate reports"
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        default='output/traffic_data.csv',
        help='Path to traffic data CSV'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/analysis',
        help='Output directory for analysis'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics only'
    )
    
    args = parser.parse_args()
    
    # Load data
    analyzer = TrafficDataAnalyzer(args.csv)
    
    if args.summary:
        analyzer.print_summary()
    else:
        analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
