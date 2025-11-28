"""
Main Pipeline for Multi-Camera Traffic Analysis System
Orchestrates detection, tracking, coordinate transformation, fusion, and visualization
"""

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO
from tqdm import tqdm

from fusion import MultiCameraFusion
from analytics import TrafficAnalytics, Zone
from vehicle import Vehicle


class CameraProcessor:
    """
    Processes video from a single camera.
    """
    
    def __init__(self, 
                 camera_id: int,
                 video_path: str,
                 model: YOLO,
                 config: dict):
        """
        Initialize camera processor.
        
        Args:
            camera_id: Camera identifier
            video_path: Path to video file
            model: YOLO model for detection
            config: Configuration dictionary
        """
        self.camera_id = camera_id
        self.video_path = video_path
        self.model = model
        self.config = config
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tracking
        self.frame_count = 0
        
        print(f"Camera {camera_id} initialized:")
        print(f"  Video: {video_path}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Read next frame from video.
        
        Returns:
            (success, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Resize if configured
            resize_width = self.config.get('performance', {}).get('resize_width')
            if resize_width and resize_width != self.width:
                aspect = self.height / self.width
                new_height = int(resize_width * aspect)
                frame = cv2.resize(frame, (resize_width, new_height))
        
        return ret, frame
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections
        """
        model_config = self.config.get('model', {})
        
        # Run YOLO inference with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=model_config.get('conf_threshold', 0.25),
            iou=model_config.get('iou_threshold', 0.45),
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box data
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Get track ID
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    track_id = i
                
                # Get class name
                class_names = self.config.get('classes', {})
                class_name = class_names.get(cls, f"Class_{cls}")
                
                detection = {
                    'bbox': box,
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': class_name,
                    'local_id': track_id
                }
                
                detections.append(detection)
        
        return detections
    
    def release(self):
        """Release video capture."""
        self.cap.release()


class MultiCameraTrafficSystem:
    """
    Main system for multi-camera traffic analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize system.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"\n{'='*60}")
        print("Multi-Camera Traffic Analysis System")
        print(f"{'='*60}\n")
        
        # Load YOLO model
        model_path = self.config['model']['path']
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ Model loaded\n")
        
        # Load global map
        map_path = self.config['global_map']['image_path']
        self.global_map = cv2.imread(map_path)
        if self.global_map is None:
            raise ValueError(f"Cannot load global map: {map_path}")
        print(f"Global map loaded: {map_path}")
        print(f"  Size: {self.global_map.shape[1]}x{self.global_map.shape[0]}\n")
        
        # Initialize cameras
        self.cameras: Dict[int, CameraProcessor] = {}
        for cam_config in self.config['cameras']:
            if cam_config.get('enabled', True):
                camera = CameraProcessor(
                    camera_id=cam_config['id'],
                    video_path=cam_config['video_path'],
                    model=self.model,
                    config=self.config
                )
                self.cameras[cam_config['id']] = camera
        
        print()
        
        # Initialize fusion system
        self.fusion = MultiCameraFusion(self.config)
        
        # Load homography matrices
        for cam_config in self.config['cameras']:
            if cam_config.get('enabled', True):
                self.fusion.load_homography_matrix(
                    cam_config['id'],
                    cam_config['homography_matrix']
                )
        
        print()
        
        # Initialize analytics
        self.analytics = TrafficAnalytics(self.config, self.global_map)
        
        # Visualization settings
        self.viz_config = self.config.get('visualization', {})
        self.colors = self.config.get('colors', {})
        
        # Output video writer
        self.output_writer = None
        if self.config.get('output', {}).get('save_visualization', False):
            self.setup_output_video()
        
        self.frame_number = 0
        self.running = True
    
    def setup_output_video(self):
        """Setup output video writer."""
        output_config = self.config.get('output', {})
        output_path = output_config.get('output_video', 'output/result.mp4')
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use global map dimensions
        h, w = self.global_map.shape[:2]
        fps = output_config.get('fps', 30)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"Output video: {output_path}")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {fps}\n")
    
    def setup_zones_interactive(self):
        """Setup zones interactively."""
        print("\n{'='*60}")
        print("Zone Setup")
        print("='*60}")
        print("Press 'a' to add zone, 's' to skip, 'q' to quit")
        
        cv2.imshow("Global Map", self.global_map)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):
                zone_name = input("Enter zone name: ")
                zone = self.analytics.zone_manager.create_zone_interactive(
                    self.global_map, zone_name
                )
                if zone:
                    self.analytics.zone_manager.add_zone(zone)
                    print(f"✓ Zone '{zone_name}' added")
            elif key == ord('s') or key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"\n✓ {len(self.analytics.zone_manager.zones)} zones configured\n")
    
    def process_frame(self) -> bool:
        """
        Process one frame from all cameras.
        
        Returns:
            True if successful, False if videos ended
        """
        camera_detections = {}
        camera_frames = {}
        
        # Read and process each camera
        for camera_id, camera in self.cameras.items():
            ret, frame = camera.read_frame()
            
            if not ret:
                return False
            
            camera_frames[camera_id] = frame
            
            # Skip frames if configured
            frame_skip = self.config.get('performance', {}).get('frame_skip', 0)
            if frame_skip > 0 and self.frame_number % (frame_skip + 1) != 0:
                camera_detections[camera_id] = []
                continue
            
            # Detect objects
            detections = camera.detect(frame)
            camera_detections[camera_id] = detections
        
        self.frame_number += 1
        
        # Fusion
        vehicles = self.fusion.process_frame(camera_detections)
        
        # Analytics
        self.analytics.update(vehicles, self.frame_number)
        
        # Visualization
        if self.viz_config.get('display_global_map', True):
            self.visualize(vehicles, camera_frames)
        
        return True
    
    def visualize(self, vehicles: List[Vehicle], camera_frames: Dict[int, np.ndarray]):
        """
        Visualize tracking results.
        
        Args:
            vehicles: List of tracked vehicles
            camera_frames: Dict of camera frames
        """
        # Create visualization on global map
        vis_map = self.global_map.copy()
        
        # Draw zones
        if self.viz_config.get('draw_zones', True):
            self.analytics.visualize(vis_map)
        
        # Draw vehicles
        trajectory_length = self.viz_config.get('trajectory_length', 50)
        point_size = self.viz_config.get('point_size', 5)
        line_thickness = self.viz_config.get('line_thickness', 2)
        
        for vehicle in vehicles:
            # Get color for class
            color = self.colors.get(vehicle.class_name, [255, 255, 255])
            color_bgr = tuple(color)  # Already in BGR
            
            # Draw trajectory
            trajectory = vehicle.get_recent_trajectory(trajectory_length)
            if len(trajectory) > 1:
                points = trajectory[:, :2].astype(np.int32)
                cv2.polylines(vis_map, [points], False, color_bgr, line_thickness)
            
            # Draw current position
            pos = (int(vehicle.current_position[0]), int(vehicle.current_position[1]))
            cv2.circle(vis_map, pos, point_size, color_bgr, -1)
            cv2.circle(vis_map, pos, point_size + 2, (0, 0, 0), 2)
            
            # Draw ID
            font_scale = self.viz_config.get('font_scale', 0.5)
            cv2.putText(vis_map, f"ID:{vehicle.global_id}",
                       (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            cv2.putText(vis_map, f"ID:{vehicle.global_id}",
                       (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 1)
        
        # Add statistics overlay
        stats = self.fusion.get_statistics()
        y_offset = 30
        cv2.putText(vis_map, f"Frame: {self.frame_number}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_map, f"Active Vehicles: {stats['active_vehicles']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_map, f"Cars: {stats['by_class']['Car']} | "
                             f"Bikes: {stats['by_class']['Bike']} | "
                             f"Pedestrians: {stats['by_class']['Pedestrian']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Global Traffic View", vis_map)
        
        # Display camera feeds if configured
        if self.viz_config.get('display_camera_feeds', False):
            for camera_id, frame in camera_frames.items():
                cv2.imshow(f"Camera {camera_id}", frame)
        
        # Save to video
        if self.output_writer:
            self.output_writer.write(vis_map)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('p'):  # Pause
            cv2.waitKey(0)
    
    def run(self):
        """Run the main processing loop."""
        print(f"{'='*60}")
        print("Starting Processing")
        print(f"{'='*60}\n")
        
        # Get total frames from first camera
        first_camera = list(self.cameras.values())[0]
        total_frames = first_camera.total_frames
        
        # Progress bar
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while self.running:
                success = self.process_frame()
                
                if not success:
                    break
                
                pbar.update(1)
        
        print(f"\n{'='*60}")
        print("Processing Complete")
        print(f"{'='*60}\n")
        
        # Print final statistics
        stats = self.fusion.get_statistics()
        print("Final Statistics:")
        print(f"  Total vehicles tracked: {stats['total_tracked']}")
        print(f"  Active vehicles: {stats['active_vehicles']}")
        print(f"  Cars: {stats['by_class']['Car']}")
        print(f"  Bikes: {stats['by_class']['Bike']}")
        print(f"  Pedestrians: {stats['by_class']['Pedestrian']}")
        
        # Zone statistics
        zone_stats = self.analytics.get_statistics()
        if zone_stats['zones']:
            print("\nZone Statistics:")
            for zone in zone_stats['zones']:
                print(f"  {zone['name']}:")
                print(f"    Current count: {zone['current_count']}")
                print(f"    Total entries: {zone['total_entries']}")
        
        print()
    
    def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up...")
        
        # Release cameras
        for camera in self.cameras.values():
            camera.release()
        
        # Release output video
        if self.output_writer:
            self.output_writer.release()
        
        # Close analytics
        self.analytics.close()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Traffic Analysis System"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--setup-zones',
        action='store_true',
        help='Setup zones interactively before processing'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = MultiCameraTrafficSystem(args.config)
        
        # Setup zones if requested
        if args.setup_zones:
            system.setup_zones_interactive()
        
        # Run processing
        system.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'system' in locals():
            system.cleanup()


if __name__ == "__main__":
    main()
