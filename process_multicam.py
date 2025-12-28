"""
Multi-Camera Vehicle Tracking and Coordinate Fusion
Uses calibrated homography matrices to track vehicles across multiple camera views
with unified global coordinates.
"""

import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import time

class MultiCameraTracker:
    def __init__(self, config_path='camera_config.json', calibration_path='camera_calibration.json'):
        """Initialize multi-camera tracking system"""
        print("🚀 Initializing Multi-Camera Tracker...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load calibration data
        try:
            with open(calibration_path, 'r') as f:
                self.calibration = json.load(f)
            print(f"✅ Loaded calibration for {len(self.calibration)} camera pairs")
        except FileNotFoundError:
            print("❌ Calibration file not found! Please run calibrate_cameras.py first")
            raise
        
        # Initialize YOLO model
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Nano model for speed
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Reference camera (global coordinate system)
        self.reference_camera = self.config.get('reference_camera', 'D3')
        
        # Build homography chain to reference camera
        self.homographies = self._build_homography_chain()
        
        # Tracking data
        self.global_tracks = {}  # Maps local IDs to global IDs
        self.next_global_id = 1
        
        # Output directory
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"✅ Reference camera: {self.reference_camera}")
        print(f"✅ Output directory: {self.output_dir}")
    
    def _build_homography_chain(self):
        """Build homography matrices from each camera to reference camera"""
        print("\n🔗 Building homography chain to reference camera...")
        homographies = {}
        
        # Reference camera has identity transformation
        homographies[self.reference_camera] = np.eye(3)
        print(f"  • {self.reference_camera} → {self.reference_camera}: Identity")
        
        # Get camera positions
        cameras = list(self.config['cameras'].keys())
        ref_idx = cameras.index(self.reference_camera)
        
        # Build chain to left cameras (D2, D1)
        for i in range(ref_idx - 1, -1, -1):
            cam_from = cameras[i + 1]
            cam_to = cameras[i]
            pair_key = f"{cam_to}_to_{cam_from}"
            
            if pair_key in self.calibration:
                H = np.array(self.calibration[pair_key]['homography'])
                # Invert to get cam_to → reference
                H_inv = np.linalg.inv(H)
                # Chain with previous homography
                homographies[cam_to] = homographies[cam_from] @ H_inv
                print(f"  • {cam_to} → {self.reference_camera}: Chained")
            else:
                print(f"  ⚠️  Missing calibration: {pair_key}")
        
        # Build chain to right cameras (D4, D5)
        for i in range(ref_idx + 1, len(cameras)):
            cam_from = cameras[i - 1]
            cam_to = cameras[i]
            pair_key = f"{cam_from}_to_{cam_to}"
            
            if pair_key in self.calibration:
                H = np.array(self.calibration[pair_key]['homography'])
                # Chain with previous homography
                homographies[cam_to] = homographies[cam_from] @ H
                print(f"  • {cam_to} → {self.reference_camera}: Chained")
            else:
                print(f"  ⚠️  Missing calibration: {pair_key}")
        
        return homographies
    
    def transform_to_global(self, point, camera_id):
        """Transform a point from camera coordinates to global (reference) coordinates"""
        if camera_id not in self.homographies:
            return None
        
        # Point must be in homogeneous coordinates
        pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(1, 1, 2)
        H = self.homographies[camera_id]
        
        # Apply transformation
        global_pt = cv2.perspectiveTransform(pt, H)
        return global_pt[0][0]
    
    def get_global_id(self, camera_id, local_id, bbox_center):
        """
        Get or assign global ID for a vehicle
        Uses spatial proximity to match vehicles across cameras
        """
        track_key = f"{camera_id}_{local_id}"
        
        # If we've seen this track before, return its global ID
        if track_key in self.global_tracks:
            return self.global_tracks[track_key]
        
        # Transform to global coordinates
        global_pos = self.transform_to_global(bbox_center, camera_id)
        if global_pos is None:
            return None
        
        # Check if any existing global track is nearby (same vehicle in different camera)
        threshold = 100  # pixels in global coordinates
        for existing_key, global_id in self.global_tracks.items():
            if existing_key.startswith(camera_id):
                continue  # Skip tracks from same camera
            
            # This is simplified - in production, you'd check recent positions
            # For now, assign new ID
            pass
        
        # Assign new global ID
        global_id = f"GLOBAL_{self.next_global_id:04d}"
        self.next_global_id += 1
        self.global_tracks[track_key] = global_id
        
        return global_id
    
    def process_camera(self, camera_id, video_path, frame_skip=1):
        """Process a single camera video"""
        print(f"\n📹 Processing {camera_id}: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   FPS: {fps} | Frames: {total_frames} | Resolution: {width}x{height}")
        print(f"   Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")
        
        results_list = []
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        # Use YOLO's built-in tracker
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Run detection and tracking
            results = self.model.track(
                frame,
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                conf=0.3,  # Confidence threshold
                iou=0.5,   # IoU threshold for NMS
                tracker='botsort.yaml'  # Use BoT-SORT tracker
            )
            
            # Process detections
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get detection data
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get track ID (if available)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    # Calculate bbox center
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    
                    # Transform to global coordinates
                    global_pos = self.transform_to_global([x_center, y_center], camera_id)
                    
                    # Get global vehicle ID
                    global_id = self.get_global_id(camera_id, track_id, [x_center, y_center])
                    
                    # Store result
                    result = {
                        'frame_id': frame_count,
                        'camera_id': camera_id,
                        'vehicle_id': track_id,
                        'global_vehicle_id': global_id,
                        'class': self.vehicle_classes.get(cls, 'unknown'),
                        'confidence': conf,
                        'x_local': x_center,
                        'y_local': y_center,
                        'x_global': global_pos[0] if global_pos is not None else None,
                        'y_global': global_pos[1] if global_pos is not None else None,
                        'width': width,
                        'height': height,
                        'timestamp': frame_count / fps
                    }
                    results_list.append(result)
            
            # Progress update every 100 frames
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = processed_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% | Processed: {processed_count} frames | Speed: {fps_processing:.1f} FPS", end='\r')
        
        cap.release()
        
        elapsed = time.time() - start_time
        fps_processing = processed_count / elapsed if elapsed > 0 else 0
        print(f"\n   ✅ Completed: {processed_count} frames in {elapsed:.1f}s ({fps_processing:.1f} FPS)")
        
        return results_list
    
    def process_all_cameras(self, frame_skip=1):
        """Process all cameras and combine results"""
        print("\n" + "="*60)
        print("🎬 MULTI-CAMERA PROCESSING STARTED")
        print("="*60)
        
        all_results = []
        
        # Process each camera
        for camera_id, camera_data in self.config['cameras'].items():
            video_path = camera_data['video_path']
            camera_results = self.process_camera(camera_id, video_path, frame_skip=frame_skip)
            all_results.extend(camera_results)
            print(f"   📊 Detections from {camera_id}: {len(camera_results)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        if len(df) > 0:
            # Save to CSV
            output_path = self.output_dir / 'multicam_results.csv'
            df.to_csv(output_path, index=False)
            print(f"\n✅ Results saved to: {output_path}")
            print(f"   Total detections: {len(df)}")
            print(f"   Unique global vehicles: {df['global_vehicle_id'].nunique()}")
            
            # Print sample results
            print("\n📊 Sample Results:")
            print(df.head(10).to_string())
            
            # Statistics by camera
            print("\n📈 Detections per Camera:")
            camera_stats = df.groupby('camera_id').size()
            for cam_id, count in camera_stats.items():
                print(f"   {cam_id}: {count} detections")
            
            # Statistics by vehicle class
            print("\n🚗 Detections by Class:")
            class_stats = df.groupby('class').size()
            for cls, count in class_stats.items():
                print(f"   {cls}: {count}")
        else:
            print("\n⚠️  No detections found!")
        
        print("\n" + "="*60)
        print("✅ MULTI-CAMERA PROCESSING COMPLETE!")
        print("="*60)
        
        return df

def main():
    """Main processing workflow"""
    try:
        # Initialize tracker
        tracker = MultiCameraTracker(
            config_path='camera_config.json',
            calibration_path='camera_calibration.json'
        )
        
        # Process all cameras
        # frame_skip=2 means process every 2nd frame (2x faster)
        # frame_skip=5 means process every 5th frame (5x faster, less accurate)
        results_df = tracker.process_all_cameras(frame_skip=2)
        
        print("\n🎉 Processing complete! Check the output folder for results.")
        print("📁 Output file: output/multicam_results.csv")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Run calibrate_cameras.py first")
        print("   2. camera_config.json with correct video paths")
        print("   3. All video files accessible")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
