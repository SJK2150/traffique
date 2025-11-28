"""
Analytics Module for Traffic Analysis
Provides zone counting, distance measurement, CSV logging, and heatmap generation
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from vehicle import Vehicle, VehicleManager


class Zone:
    """
    Represents a polygon zone for counting vehicles.
    """
    
    def __init__(self, 
                 name: str, 
                 polygon: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 255)):
        """
        Initialize zone.
        
        Args:
            name: Zone name
            polygon: Array of polygon vertices (N, 2)
            color: RGB color for visualization
        """
        self.name = name
        self.polygon = polygon.astype(np.float32)
        self.color = color
        self.vehicle_counts = {
            'Car': 0,
            'Bike': 0,
            'Pedestrian': 0,
            'total': 0
        }
        self.vehicles_in_zone = set()  # Set of vehicle IDs currently in zone
        self.total_entries = 0
        
    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the zone.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            True if point is inside zone
        """
        result = cv2.pointPolygonTest(
            self.polygon,
            (float(point[0]), float(point[1])),
            False
        )
        return result >= 0
    
    def update(self, vehicles: List[Vehicle]):
        """
        Update zone counts with current vehicles.
        
        Args:
            vehicles: List of active vehicles
        """
        current_vehicles = set()
        
        # Reset counts
        self.vehicle_counts = {
            'Car': 0,
            'Bike': 0,
            'Pedestrian': 0,
            'total': 0
        }
        
        # Count vehicles currently in zone
        for vehicle in vehicles:
            if self.is_point_inside(vehicle.current_position):
                current_vehicles.add(vehicle.global_id)
                self.vehicle_counts[vehicle.class_name] += 1
                self.vehicle_counts['total'] += 1
        
        # Track new entries
        new_entries = current_vehicles - self.vehicles_in_zone
        self.total_entries += len(new_entries)
        
        self.vehicles_in_zone = current_vehicles
    
    def draw(self, image: np.ndarray, alpha: float = 0.3):
        """
        Draw zone on image.
        
        Args:
            image: Image to draw on
            alpha: Transparency factor
        """
        # Create overlay
        overlay = image.copy()
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [self.polygon.astype(np.int32)], self.color)
        
        # Blend with original
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw border
        cv2.polylines(image, [self.polygon.astype(np.int32)], True, self.color, 2)
        
        # Draw label
        centroid = self.polygon.mean(axis=0).astype(np.int32)
        label = f"{self.name}: {self.vehicle_counts['total']}"
        
        # Text background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            image,
            (centroid[0] - 5, centroid[1] - text_height - 5),
            (centroid[0] + text_width + 5, centroid[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            image, label,
            (centroid[0], centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
    
    def get_statistics(self) -> dict:
        """Get zone statistics."""
        return {
            'name': self.name,
            'current_count': self.vehicle_counts['total'],
            'cars': self.vehicle_counts['Car'],
            'bikes': self.vehicle_counts['Bike'],
            'pedestrians': self.vehicle_counts['Pedestrian'],
            'total_entries': self.total_entries
        }


class ZoneManager:
    """
    Manages multiple zones.
    """
    
    def __init__(self):
        """Initialize zone manager."""
        self.zones: List[Zone] = []
    
    def add_zone(self, zone: Zone):
        """Add a zone."""
        self.zones.append(zone)
    
    def create_zone_interactive(self, 
                               map_image: np.ndarray,
                               zone_name: str) -> Optional[Zone]:
        """
        Create zone by clicking points on map.
        
        Args:
            map_image: Global map image
            zone_name: Name for the zone
            
        Returns:
            Created zone or None if cancelled
        """
        points = []
        display = map_image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                display = map_image.copy()
                
                # Draw points
                for i, pt in enumerate(points):
                    cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
                    cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw lines
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(display, tuple(points[i]), tuple(points[i+1]), 
                               (0, 255, 0), 2)
                
                # Draw closing line if 3+ points
                if len(points) >= 3:
                    cv2.line(display, tuple(points[-1]), tuple(points[0]), 
                           (0, 255, 0), 2)
        
        window_name = f"Create Zone: {zone_name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print(f"\nCreating zone: {zone_name}")
        print("Click to add points. Press 's' to save, 'r' to reset, 'q' to cancel")
        
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(points) >= 3:
                cv2.destroyWindow(window_name)
                polygon = np.array(points, dtype=np.float32)
                return Zone(zone_name, polygon)
            elif key == ord('r'):
                points = []
                display = map_image.copy()
            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
    
    def update_all(self, vehicles: List[Vehicle]):
        """Update all zones."""
        for zone in self.zones:
            zone.update(vehicles)
    
    def draw_all(self, image: np.ndarray):
        """Draw all zones on image."""
        for zone in self.zones:
            zone.draw(image)
    
    def get_all_statistics(self) -> List[dict]:
        """Get statistics for all zones."""
        return [zone.get_statistics() for zone in self.zones]


class DistanceMeasurement:
    """
    Tool for measuring real-world distances on global map.
    """
    
    def __init__(self, pixels_per_meter: float = 1.0):
        """
        Initialize distance measurement tool.
        
        Args:
            pixels_per_meter: Scale factor (pixels per meter)
        """
        self.pixels_per_meter = pixels_per_meter
    
    def measure_distance(self, 
                        point1: Tuple[float, float],
                        point2: Tuple[float, float]) -> float:
        """
        Measure distance between two points in meters.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance in meters
        """
        pixel_distance = np.sqrt(
            (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2
        )
        return pixel_distance / self.pixels_per_meter
    
    def measure_interactive(self, map_image: np.ndarray) -> Optional[float]:
        """
        Measure distance interactively by clicking two points.
        
        Args:
            map_image: Global map image
            
        Returns:
            Measured distance in meters or None if cancelled
        """
        points = []
        display = map_image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display
            
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append([x, y])
                display = map_image.copy()
                
                # Draw points
                for pt in points:
                    cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
                
                # Draw line
                if len(points) == 2:
                    cv2.line(display, tuple(points[0]), tuple(points[1]), 
                           (0, 0, 255), 2)
                    
                    distance = self.measure_distance(
                        tuple(points[0]), tuple(points[1])
                    )
                    
                    mid_x = (points[0][0] + points[1][0]) // 2
                    mid_y = (points[0][1] + points[1][1]) // 2
                    
                    cv2.putText(display, f"{distance:.2f}m",
                              (mid_x, mid_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.namedWindow("Distance Measurement")
        cv2.setMouseCallback("Distance Measurement", mouse_callback)
        
        print("\nDistance Measurement")
        print("Click two points. Press 's' to save, 'r' to reset, 'q' to cancel")
        
        result = None
        
        while True:
            cv2.imshow("Distance Measurement", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(points) == 2:
                result = self.measure_distance(tuple(points[0]), tuple(points[1]))
                break
            elif key == ord('r'):
                points = []
                display = map_image.copy()
            elif key == ord('q'):
                break
        
        cv2.destroyWindow("Distance Measurement")
        return result


class CSVLogger:
    """
    Logs vehicle data to CSV file.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize CSV logger.
        
        Args:
            output_path: Path to output CSV file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.data_buffer = []
        self.buffer_size = 100  # Write every N frames
        
        # Initialize CSV with headers
        self.initialize_csv()
    
    def initialize_csv(self):
        """Create CSV file with headers."""
        headers = [
            'timestamp', 'frame', 'vehicle_id', 'class', 
            'global_x', 'global_y', 'camera_id',
            'confidence', 'total_distance', 'average_speed',
            'trajectory_length'
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.output_path, index=False)
        
        print(f"✓ CSV logger initialized: {self.output_path}")
    
    def log_frame(self, vehicles: List[Vehicle], frame_number: int):
        """
        Log vehicles for current frame.
        
        Args:
            vehicles: List of vehicles to log
            frame_number: Current frame number
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        for vehicle in vehicles:
            row = {
                'timestamp': timestamp,
                'frame': frame_number,
                'vehicle_id': vehicle.global_id,
                'class': vehicle.class_name,
                'global_x': vehicle.current_position[0],
                'global_y': vehicle.current_position[1],
                'camera_id': vehicle.current_camera_id,
                'confidence': vehicle.confidence,
                'total_distance': vehicle.total_distance,
                'average_speed': vehicle.average_speed,
                'trajectory_length': len(vehicle.trajectory)
            }
            self.data_buffer.append(row)
        
        # Write buffer if full
        if len(self.data_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered data to CSV."""
        if not self.data_buffer:
            return
        
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.output_path, mode='a', header=False, index=False)
        self.data_buffer.clear()
    
    def close(self):
        """Flush and close logger."""
        self.flush()


class HeatmapGenerator:
    """
    Generates heatmap of vehicle positions.
    """
    
    def __init__(self, map_shape: Tuple[int, int]):
        """
        Initialize heatmap generator.
        
        Args:
            map_shape: Shape of global map (height, width)
        """
        self.heatmap = np.zeros(map_shape, dtype=np.float32)
        self.map_shape = map_shape
    
    def add_vehicle(self, position: Tuple[float, float], weight: float = 1.0):
        """
        Add vehicle position to heatmap.
        
        Args:
            position: (x, y) position
            weight: Weight for this position
        """
        x, y = int(position[0]), int(position[1])
        
        # Check bounds
        if 0 <= y < self.map_shape[0] and 0 <= x < self.map_shape[1]:
            # Add Gaussian blob
            sigma = 10
            for dy in range(-sigma*2, sigma*2+1):
                for dx in range(-sigma*2, sigma*2+1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]:
                        distance = np.sqrt(dx**2 + dy**2)
                        value = weight * np.exp(-(distance**2) / (2 * sigma**2))
                        self.heatmap[ny, nx] += value
    
    def add_vehicles(self, vehicles: List[Vehicle]):
        """Add multiple vehicles to heatmap."""
        for vehicle in vehicles:
            self.add_vehicle(vehicle.current_position)
    
    def generate(self, background: np.ndarray) -> np.ndarray:
        """
        Generate heatmap visualization.
        
        Args:
            background: Background image
            
        Returns:
            Heatmap overlaid on background
        """
        # Normalize heatmap
        if self.heatmap.max() > 0:
            normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        else:
            normalized = self.heatmap.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Blend with background
        alpha = 0.5
        result = cv2.addWeighted(background, 1-alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def save(self, filepath: str, background: np.ndarray):
        """Save heatmap to file."""
        result = self.generate(background)
        cv2.imwrite(filepath, result)
        print(f"✓ Heatmap saved: {filepath}")


class TrafficAnalytics:
    """
    Main analytics controller.
    """
    
    def __init__(self, config: dict, map_image: np.ndarray):
        """
        Initialize analytics system.
        
        Args:
            config: Configuration dictionary
            map_image: Global map image
        """
        self.config = config
        self.map_image = map_image
        
        analytics_config = config.get('analytics', {})
        
        # Initialize components
        self.zone_manager = ZoneManager()
        self.distance_tool = DistanceMeasurement(
            pixels_per_meter=config.get('global_map', {}).get('scale', 1.0)
        )
        self.csv_logger = CSVLogger(
            analytics_config.get('csv_output', 'output/traffic_data.csv')
        )
        
        if analytics_config.get('enable_heatmap', True):
            self.heatmap = HeatmapGenerator(map_image.shape[:2])
        else:
            self.heatmap = None
    
    def update(self, vehicles: List[Vehicle], frame_number: int):
        """
        Update analytics for current frame.
        
        Args:
            vehicles: List of active vehicles
            frame_number: Current frame number
        """
        # Update zones
        self.zone_manager.update_all(vehicles)
        
        # Log to CSV
        if frame_number % self.config.get('analytics', {}).get('log_interval', 1) == 0:
            self.csv_logger.log_frame(vehicles, frame_number)
        
        # Update heatmap
        if self.heatmap:
            self.heatmap.add_vehicles(vehicles)
    
    def visualize(self, image: np.ndarray):
        """Draw analytics on image."""
        self.zone_manager.draw_all(image)
    
    def get_statistics(self) -> dict:
        """Get all analytics statistics."""
        return {
            'zones': self.zone_manager.get_all_statistics()
        }
    
    def close(self):
        """Close analytics and save final data."""
        self.csv_logger.close()
        
        if self.heatmap:
            heatmap_path = self.config.get('analytics', {}).get(
                'heatmap_output', 'output/heatmap.png'
            )
            self.heatmap.save(heatmap_path, self.map_image)
