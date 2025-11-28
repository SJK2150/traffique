"""
Vehicle Class for Multi-Camera Traffic Analysis
Maintains unique ID, trajectory history, and global coordinates
"""

import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


class Vehicle:
    """
    Represents a tracked vehicle across multiple cameras with global coordinate system.
    """
    
    # Class variable for generating unique IDs
    _id_counter = 0
    
    def __init__(self, 
                 global_id: int,
                 class_name: str,
                 initial_position: Tuple[float, float],
                 camera_id: int,
                 local_id: int,
                 confidence: float,
                 frame_number: int,
                 timestamp: datetime = None):
        """
        Initialize a Vehicle object.
        
        Args:
            global_id: Unique global identifier
            class_name: Vehicle type ('Car', 'Bike', 'Pedestrian')
            initial_position: (x, y) in global coordinates
            camera_id: ID of the camera that first detected this vehicle
            local_id: Tracker ID from the camera
            confidence: Detection confidence score
            frame_number: Frame number when first detected
            timestamp: Detection timestamp
        """
        self.global_id = global_id
        self.class_name = class_name
        self.camera_id = camera_id
        self.local_id = local_id
        self.confidence = confidence
        self.first_seen_frame = frame_number
        self.last_seen_frame = frame_number
        self.timestamp = timestamp or datetime.now()
        
        # Trajectory history - list of (x, y, frame, camera_id, confidence)
        self.trajectory: List[Tuple[float, float, int, int, float]] = []
        self.trajectory.append((initial_position[0], initial_position[1], 
                               frame_number, camera_id, confidence))
        
        # Current state
        self.current_position = initial_position
        self.current_camera_id = camera_id
        self.is_active = True
        self.frames_since_update = 0
        
        # Statistics
        self.total_distance = 0.0
        self.average_speed = 0.0
        self.max_speed = 0.0
        
    @classmethod
    def generate_id(cls) -> int:
        """Generate a unique vehicle ID."""
        cls._id_counter += 1
        return cls._id_counter
    
    def update(self, 
               position: Tuple[float, float],
               camera_id: int,
               local_id: int,
               confidence: float,
               frame_number: int):
        """
        Update vehicle with new detection.
        
        Args:
            position: New (x, y) position in global coordinates
            camera_id: Camera that detected the vehicle
            local_id: Local tracker ID
            confidence: Detection confidence
            frame_number: Current frame number
        """
        # Calculate distance moved
        if len(self.trajectory) > 0:
            prev_x, prev_y, prev_frame, _, _ = self.trajectory[-1]
            distance = np.sqrt((position[0] - prev_x)**2 + (position[1] - prev_y)**2)
            self.total_distance += distance
            
            # Calculate speed (distance per frame)
            frame_diff = frame_number - prev_frame
            if frame_diff > 0:
                speed = distance / frame_diff
                self.average_speed = (self.average_speed * len(self.trajectory) + speed) / (len(self.trajectory) + 1)
                self.max_speed = max(self.max_speed, speed)
        
        # Add to trajectory
        self.trajectory.append((position[0], position[1], frame_number, camera_id, confidence))
        
        # Update state
        self.current_position = position
        self.current_camera_id = camera_id
        self.local_id = local_id
        self.confidence = confidence
        self.last_seen_frame = frame_number
        self.frames_since_update = 0
        self.is_active = True
    
    def mark_lost(self):
        """Mark vehicle as lost (not detected in current frame)."""
        self.frames_since_update += 1
        
    def should_remove(self, max_frames_lost: int = 30) -> bool:
        """
        Check if vehicle should be removed from tracking.
        
        Args:
            max_frames_lost: Maximum frames without detection before removal
            
        Returns:
            True if vehicle should be removed
        """
        return self.frames_since_update > max_frames_lost
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        Get trajectory as numpy array.
        
        Returns:
            Array of shape (N, 5) with columns [x, y, frame, camera_id, confidence]
        """
        return np.array(self.trajectory)
    
    def get_recent_trajectory(self, n_points: int = 50) -> np.ndarray:
        """
        Get recent trajectory points.
        
        Args:
            n_points: Number of recent points to return
            
        Returns:
            Array of recent trajectory points
        """
        if len(self.trajectory) <= n_points:
            return self.get_trajectory_array()
        return np.array(self.trajectory[-n_points:])
    
    def is_in_zone(self, zone_polygon: np.ndarray) -> bool:
        """
        Check if vehicle's current position is inside a zone polygon.
        
        Args:
            zone_polygon: Polygon vertices as array of shape (N, 2)
            
        Returns:
            True if vehicle is inside the zone
        """
        import cv2
        point = np.array([self.current_position], dtype=np.float32)
        result = cv2.pointPolygonTest(zone_polygon, 
                                     (float(self.current_position[0]), 
                                      float(self.current_position[1])), 
                                     False)
        return result >= 0
    
    def get_velocity_vector(self, n_frames: int = 5) -> Tuple[float, float]:
        """
        Calculate velocity vector from recent trajectory.
        
        Args:
            n_frames: Number of frames to look back
            
        Returns:
            (vx, vy) velocity vector
        """
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        
        # Get recent points
        recent = self.get_recent_trajectory(n_frames)
        if len(recent) < 2:
            return (0.0, 0.0)
        
        # Calculate average velocity
        start_x, start_y = recent[0, 0], recent[0, 1]
        end_x, end_y = recent[-1, 0], recent[-1, 1]
        frame_diff = recent[-1, 2] - recent[0, 2]
        
        if frame_diff == 0:
            return (0.0, 0.0)
        
        vx = (end_x - start_x) / frame_diff
        vy = (end_y - start_y) / frame_diff
        
        return (vx, vy)
    
    def to_dict(self) -> dict:
        """
        Convert vehicle data to dictionary for CSV export.
        
        Returns:
            Dictionary with vehicle information
        """
        return {
            'global_id': self.global_id,
            'class': self.class_name,
            'x': self.current_position[0],
            'y': self.current_position[1],
            'camera_id': self.current_camera_id,
            'frame': self.last_seen_frame,
            'confidence': self.confidence,
            'total_distance': self.total_distance,
            'average_speed': self.average_speed,
            'trajectory_length': len(self.trajectory)
        }
    
    def __repr__(self) -> str:
        return (f"Vehicle(id={self.global_id}, class={self.class_name}, "
                f"pos={self.current_position}, camera={self.current_camera_id}, "
                f"trajectory_len={len(self.trajectory)})")


class VehicleManager:
    """
    Manages all tracked vehicles across multiple cameras.
    """
    
    def __init__(self, max_frames_lost: int = 30):
        """
        Initialize VehicleManager.
        
        Args:
            max_frames_lost: Maximum frames without detection before removing vehicle
        """
        self.vehicles: List[Vehicle] = []
        self.max_frames_lost = max_frames_lost
        self.removed_vehicles: List[Vehicle] = []
        
    def add_vehicle(self, vehicle: Vehicle):
        """Add a new vehicle to tracking."""
        self.vehicles.append(vehicle)
    
    def get_vehicle_by_id(self, global_id: int) -> Optional[Vehicle]:
        """Get vehicle by global ID."""
        for vehicle in self.vehicles:
            if vehicle.global_id == global_id:
                return vehicle
        return None
    
    def get_active_vehicles(self) -> List[Vehicle]:
        """Get all currently active vehicles."""
        return [v for v in self.vehicles if v.is_active]
    
    def get_vehicles_in_zone(self, zone_polygon: np.ndarray) -> List[Vehicle]:
        """Get all active vehicles inside a zone."""
        return [v for v in self.get_active_vehicles() if v.is_in_zone(zone_polygon)]
    
    def get_vehicles_by_class(self, class_name: str) -> List[Vehicle]:
        """Get all active vehicles of a specific class."""
        return [v for v in self.get_active_vehicles() if v.class_name == class_name]
    
    def update_lost_vehicles(self):
        """Mark vehicles as lost and remove old ones."""
        to_remove = []
        
        for vehicle in self.vehicles:
            if vehicle.should_remove(self.max_frames_lost):
                to_remove.append(vehicle)
        
        for vehicle in to_remove:
            self.vehicles.remove(vehicle)
            self.removed_vehicles.append(vehicle)
    
    def get_statistics(self) -> dict:
        """Get overall tracking statistics."""
        active = self.get_active_vehicles()
        return {
            'total_vehicles': len(self.vehicles),
            'active_vehicles': len(active),
            'removed_vehicles': len(self.removed_vehicles),
            'total_tracked': len(self.vehicles) + len(self.removed_vehicles),
            'by_class': {
                'Car': len([v for v in active if v.class_name == 'Car']),
                'Bike': len([v for v in active if v.class_name == 'Bike']),
                'Pedestrian': len([v for v in active if v.class_name == 'Pedestrian'])
            }
        }
    
    def clear(self):
        """Clear all vehicles."""
        self.vehicles.clear()
        self.removed_vehicles.clear()
        Vehicle._id_counter = 0
