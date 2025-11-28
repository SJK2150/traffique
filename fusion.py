"""
Fusion Module for Multi-Camera Traffic Analysis
Merges detections from multiple cameras based on global coordinate proximity
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from vehicle import Vehicle, VehicleManager


class CoordinateFusion:
    """
    Fuses detections from multiple cameras using global coordinates.
    """
    
    def __init__(self, 
                 distance_threshold: float = 2.0,
                 frame_window: int = 5,
                 confidence_weight: float = 0.3):
        """
        Initialize coordinate fusion system.
        
        Args:
            distance_threshold: Maximum distance (meters) to consider vehicles the same
            frame_window: Number of frames to look back for potential matches
            confidence_weight: Weight of confidence in fusion decision (0-1)
        """
        self.distance_threshold = distance_threshold
        self.frame_window = frame_window
        self.confidence_weight = confidence_weight
        
        # Track recent detections for temporal fusion
        self.recent_detections: Dict[int, List[Dict]] = {}  # camera_id -> list of detections
        
    def calculate_distance(self, 
                          pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Distance in same units as coordinates
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_matching_vehicle(self,
                             detection: Dict,
                             vehicle_manager: VehicleManager) -> Optional[Vehicle]:
        """
        Find existing vehicle that matches the detection.
        
        Args:
            detection: Detection dict with keys: position, class_name, confidence, camera_id
            vehicle_manager: Manager containing tracked vehicles
            
        Returns:
            Matching vehicle or None
        """
        position = detection['position']
        class_name = detection['class_name']
        confidence = detection['confidence']
        camera_id = detection['camera_id']
        
        best_match = None
        best_score = float('inf')
        
        # Get active vehicles of same class
        candidates = [v for v in vehicle_manager.get_active_vehicles() 
                     if v.class_name == class_name]
        
        for vehicle in candidates:
            # Calculate spatial distance
            distance = self.calculate_distance(position, vehicle.current_position)
            
            if distance > self.distance_threshold:
                continue
            
            # Calculate velocity-based prediction
            vx, vy = vehicle.get_velocity_vector(n_frames=3)
            frames_diff = detection.get('frame', 0) - vehicle.last_seen_frame
            predicted_x = vehicle.current_position[0] + vx * frames_diff
            predicted_y = vehicle.current_position[1] + vy * frames_diff
            predicted_distance = self.calculate_distance(position, (predicted_x, predicted_y))
            
            # Combined score (lower is better)
            # Weight actual distance more than predicted distance
            spatial_score = 0.7 * distance + 0.3 * predicted_distance
            
            # Confidence bonus (higher confidence -> lower score)
            confidence_factor = 1.0 - (self.confidence_weight * confidence)
            
            # Camera transition bonus (if different camera, slightly prefer)
            camera_bonus = 0.9 if camera_id != vehicle.current_camera_id else 1.0
            
            final_score = spatial_score * confidence_factor * camera_bonus
            
            if final_score < best_score:
                best_score = final_score
                best_match = vehicle
        
        return best_match
    
    def merge_detections(self,
                        camera_detections: Dict[int, List[Dict]],
                        vehicle_manager: VehicleManager,
                        frame_number: int) -> List[Vehicle]:
        """
        Merge detections from all cameras.
        
        Args:
            camera_detections: Dict mapping camera_id -> list of detections
                Each detection: {position, class_name, confidence, bbox, local_id}
            vehicle_manager: Manager for tracked vehicles
            frame_number: Current frame number
            
        Returns:
            List of updated/created vehicles
        """
        # Mark all vehicles as lost initially
        for vehicle in vehicle_manager.get_active_vehicles():
            vehicle.mark_lost()
        
        updated_vehicles = []
        
        # Process detections from each camera
        for camera_id, detections in camera_detections.items():
            for detection in detections:
                detection['camera_id'] = camera_id
                detection['frame'] = frame_number
                
                # Try to match with existing vehicle
                matched_vehicle = self.find_matching_vehicle(detection, vehicle_manager)
                
                if matched_vehicle is not None:
                    # Update existing vehicle
                    matched_vehicle.update(
                        position=detection['position'],
                        camera_id=camera_id,
                        local_id=detection['local_id'],
                        confidence=detection['confidence'],
                        frame_number=frame_number
                    )
                    updated_vehicles.append(matched_vehicle)
                else:
                    # Create new vehicle
                    new_vehicle = Vehicle(
                        global_id=Vehicle.generate_id(),
                        class_name=detection['class_name'],
                        initial_position=detection['position'],
                        camera_id=camera_id,
                        local_id=detection['local_id'],
                        confidence=detection['confidence'],
                        frame_number=frame_number
                    )
                    vehicle_manager.add_vehicle(new_vehicle)
                    updated_vehicles.append(new_vehicle)
        
        # Store recent detections for temporal fusion
        self.recent_detections = camera_detections.copy()
        
        # Remove old vehicles
        vehicle_manager.update_lost_vehicles()
        
        return updated_vehicles
    
    def resolve_conflicts(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """
        Resolve conflicts where multiple vehicles are too close to each other.
        
        Args:
            vehicles: List of vehicles to check
            
        Returns:
            List of vehicles with conflicts resolved
        """
        if len(vehicles) < 2:
            return vehicles
        
        # Build position matrix
        positions = np.array([v.current_position for v in vehicles])
        
        # Calculate pairwise distances
        distances = cdist(positions, positions)
        
        # Find pairs that are too close
        to_merge = []
        n = len(vehicles)
        
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] < self.distance_threshold * 0.5:  # Very close
                    if vehicles[i].class_name == vehicles[j].class_name:
                        to_merge.append((i, j))
        
        # Merge vehicles (keep the one with longer trajectory)
        merged_indices = set()
        
        for i, j in to_merge:
            if i in merged_indices or j in merged_indices:
                continue
            
            # Keep vehicle with longer trajectory or higher confidence
            if len(vehicles[i].trajectory) >= len(vehicles[j].trajectory):
                keep, remove = i, j
            else:
                keep, remove = j, i
            
            # Merge trajectories (could be more sophisticated)
            vehicles[keep].confidence = max(vehicles[keep].confidence, 
                                           vehicles[remove].confidence)
            merged_indices.add(remove)
        
        # Return vehicles that weren't merged away
        return [v for i, v in enumerate(vehicles) if i not in merged_indices]
    
    def cross_camera_tracking(self,
                             vehicle: Vehicle,
                             camera_detections: Dict[int, List[Dict]],
                             current_frame: int) -> Optional[Dict]:
        """
        Try to find vehicle in other cameras based on predicted position.
        
        Args:
            vehicle: Vehicle to track
            camera_detections: Current detections from all cameras
            current_frame: Current frame number
            
        Returns:
            Matching detection from another camera, or None
        """
        # Predict next position
        vx, vy = vehicle.get_velocity_vector(n_frames=5)
        frames_since = current_frame - vehicle.last_seen_frame
        
        predicted_x = vehicle.current_position[0] + vx * frames_since
        predicted_y = vehicle.current_position[1] + vy * frames_since
        predicted_pos = (predicted_x, predicted_y)
        
        best_match = None
        best_distance = float('inf')
        
        # Check detections from other cameras
        for camera_id, detections in camera_detections.items():
            if camera_id == vehicle.current_camera_id:
                continue
            
            for detection in detections:
                if detection['class_name'] != vehicle.class_name:
                    continue
                
                distance = self.calculate_distance(predicted_pos, detection['position'])
                
                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = detection.copy()
                    best_match['camera_id'] = camera_id
        
        return best_match


class MultiCameraFusion:
    """
    High-level fusion manager for multi-camera system.
    """
    
    def __init__(self, config: dict):
        """
        Initialize multi-camera fusion system.
        
        Args:
            config: Configuration dictionary
        """
        fusion_config = config.get('fusion', {})
        
        self.coordinate_fusion = CoordinateFusion(
            distance_threshold=fusion_config.get('distance_threshold', 2.0),
            frame_window=fusion_config.get('frame_window', 5),
            confidence_weight=fusion_config.get('confidence_weight', 0.3)
        )
        
        self.vehicle_manager = VehicleManager(
            max_frames_lost=config.get('tracking', {}).get('track_buffer', 30)
        )
        
        self.homography_matrices: Dict[int, np.ndarray] = {}
        self.frame_count = 0
        
    def load_homography_matrix(self, camera_id: int, matrix_path: str):
        """
        Load homography matrix for a camera.
        
        Args:
            camera_id: Camera identifier
            matrix_path: Path to .npy file with homography matrix
        """
        H = np.load(matrix_path)
        self.homography_matrices[camera_id] = H
        print(f"✓ Loaded homography matrix for camera {camera_id}")
    
    def transform_to_global(self,
                           point: Tuple[float, float],
                           camera_id: int) -> Tuple[float, float]:
        """
        Transform camera coordinates to global coordinates.
        
        Args:
            point: (x, y) in camera coordinates
            camera_id: Camera identifier
            
        Returns:
            (x, y) in global coordinates
        """
        if camera_id not in self.homography_matrices:
            raise ValueError(f"No homography matrix for camera {camera_id}")
        
        H = self.homography_matrices[camera_id]
        
        # Convert to homogeneous coordinates
        pt = np.array([point[0], point[1], 1.0])
        
        # Transform
        transformed = H @ pt
        transformed = transformed / transformed[2]
        
        return (transformed[0], transformed[1])
    
    def batch_transform_to_global(self,
                                  points: np.ndarray,
                                  camera_id: int) -> np.ndarray:
        """
        Transform multiple points to global coordinates.
        
        Args:
            points: Array of shape (N, 2)
            camera_id: Camera identifier
            
        Returns:
            Transformed points array of shape (N, 2)
        """
        if camera_id not in self.homography_matrices:
            raise ValueError(f"No homography matrix for camera {camera_id}")
        
        H = self.homography_matrices[camera_id]
        
        # Add homogeneous coordinate
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])
        
        # Transform
        transformed = (H @ points_h.T).T
        
        # Normalize
        transformed = transformed / transformed[:, 2:3]
        
        return transformed[:, :2]
    
    def process_frame(self, camera_detections: Dict[int, List[Dict]]) -> List[Vehicle]:
        """
        Process detections from all cameras for current frame.
        
        Args:
            camera_detections: Dict mapping camera_id -> list of detections
                Each detection must have: bbox, class_name, confidence, local_id
                
        Returns:
            List of updated vehicles
        """
        self.frame_count += 1
        
        # Transform all detections to global coordinates
        transformed_detections = {}
        
        for camera_id, detections in camera_detections.items():
            transformed = []
            
            for det in detections:
                # Get bottom-center of bounding box
                bbox = det['bbox']  # (x1, y1, x2, y2)
                bottom_center_x = (bbox[0] + bbox[2]) / 2
                bottom_center_y = bbox[3]  # Bottom of box
                
                # Transform to global coordinates
                global_pos = self.transform_to_global(
                    (bottom_center_x, bottom_center_y),
                    camera_id
                )
                
                transformed.append({
                    'position': global_pos,
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'local_id': det['local_id'],
                    'bbox': bbox
                })
            
            transformed_detections[camera_id] = transformed
        
        # Merge detections
        updated_vehicles = self.coordinate_fusion.merge_detections(
            transformed_detections,
            self.vehicle_manager,
            self.frame_count
        )
        
        # Resolve conflicts
        updated_vehicles = self.coordinate_fusion.resolve_conflicts(updated_vehicles)
        
        return updated_vehicles
    
    def get_all_vehicles(self) -> List[Vehicle]:
        """Get all tracked vehicles."""
        return self.vehicle_manager.vehicles
    
    def get_active_vehicles(self) -> List[Vehicle]:
        """Get all active vehicles."""
        return self.vehicle_manager.get_active_vehicles()
    
    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        return self.vehicle_manager.get_statistics()
    
    def reset(self):
        """Reset the fusion system."""
        self.vehicle_manager.clear()
        self.frame_count = 0
