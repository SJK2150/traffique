"""
Multi-Camera Calibration Tool
Helps you select matching points between adjacent camera views to compute homography matrices.
"""

import cv2
import numpy as np
import json
from pathlib import Path

class CameraCalibrator:
    def __init__(self, config_path='camera_config.json'):
        """Initialize calibrator with camera configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.calibration_data = {}
        self.points = {'src': [], 'dst': []}
        self.current_camera = None
        self.neighbor_camera = None
        self.scale1 = 1.0
        self.scale2 = 1.0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select calibration points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if param['type'] == 'src':
                # Store original coordinates (unscaled)
                orig_x = x / param['scale']
                orig_y = y / param['scale']
                self.points['src'].append([orig_x, orig_y])
                print(f"  Source point {len(self.points['src'])}: ({int(orig_x)}, {int(orig_y)}) [display: ({x}, {y})]")
            else:
                # Store original coordinates (unscaled)
                orig_x = x / param['scale']
                orig_y = y / param['scale']
                self.points['dst'].append([orig_x, orig_y])
                print(f"  Target point {len(self.points['dst'])}: ({int(orig_x)}, {int(orig_y)}) [display: ({x}, {y})]")
    
    def calibrate_pair(self, cam1_id, cam2_id):
        """Calibrate a pair of adjacent cameras"""
        print(f"\n{'='*60}")
        print(f"Calibrating: {cam1_id} → {cam2_id}")
        print(f"{'='*60}")
        
        # Get video paths
        cam1_path = self.config['cameras'][cam1_id]['video_path']
        cam2_path = self.config['cameras'][cam2_id]['video_path']
        
        # Open videos and get first frame
        cap1 = cv2.VideoCapture(cam1_path)
        cap2 = cv2.VideoCapture(cam2_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("❌ Error reading video frames!")
            return None
        
        cap1.release()
        cap2.release()
        
        # Resize frames for display (maintain aspect ratio)
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        display_width = 960
        self.scale1 = display_width / w1
        self.scale2 = display_width / w2
        
        frame1_display = cv2.resize(frame1, (display_width, int(h1 * self.scale1)))
        frame2_display = cv2.resize(frame2, (display_width, int(h2 * self.scale2)))
        
        # Reset points
        self.points = {'src': [], 'dst': []}
        
        # Instructions
        print("\n📋 INSTRUCTIONS:")
        print("1. LEFT window: Click on distinctive points (e.g., road markings, corners)")
        print("2. RIGHT window: Click on the SAME points in matching order")
        print("3. Select at least 4 matching point pairs (more is better for accuracy)")
        print("4. Press 'c' when done selecting points")
        print("5. Press 'r' to reset and start over")
        print("6. Press 'q' to skip this pair")
        print("\n💡 TIP: Choose points in the OVERLAPPING region between cameras")
        print("   Good choices: Road lane markings, intersections, building corners\n")
        
        # Create windows
        cv2.namedWindow(f'{cam1_id} (Source)', cv2.WINDOW_NORMAL)
        cv2.namedWindow(f'{cam2_id} (Target)', cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback(f'{cam1_id} (Source)', self.mouse_callback, {'type': 'src', 'scale': self.scale1})
        cv2.setMouseCallback(f'{cam2_id} (Target)', self.mouse_callback, {'type': 'dst', 'scale': self.scale2})
        
        while True:
            # Draw points on frames
            display1 = frame1_display.copy()
            display2 = frame2_display.copy()
            
            # Draw source points (green circles with numbers)
            for i, pt in enumerate(self.points['src']):
                x, y = int(pt[0] * self.scale1), int(pt[1] * self.scale1)
                cv2.circle(display1, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(display1, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display1, str(i+1), (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw target points (blue circles with numbers)
            for i, pt in enumerate(self.points['dst']):
                x, y = int(pt[0] * self.scale2), int(pt[1] * self.scale2)
                cv2.circle(display2, (x, y), 8, (255, 0, 0), -1)
                cv2.circle(display2, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display2, str(i+1), (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add status text
            status = f"Points: {len(self.points['src'])} source, {len(self.points['dst'])} target | Press 'c' to compute, 'r' to reset, 'q' to skip"
            cv2.putText(display1, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(f'{cam1_id} (Source)', display1)
            cv2.imshow(f'{cam2_id} (Target)', display2)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Compute homography
                if len(self.points['src']) >= 4 and len(self.points['src']) == len(self.points['dst']):
                    # Points are already in original coordinates (stored unscaled in mouse callback)
                    src_pts = np.array(self.points['src'], dtype=np.float32)
                    dst_pts = np.array(self.points['dst'], dtype=np.float32)
                    
                    # Compute homography matrix
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        inliers = np.sum(mask)
                        print(f"\n✅ Homography computed successfully!")
                        print(f"   Inliers: {inliers}/{len(src_pts)} points")
                        print(f"   Matrix shape: {H.shape}")
                        
                        # Test the homography
                        test_pt = src_pts[0].reshape(1, 1, 2)
                        transformed = cv2.perspectiveTransform(test_pt, H)
                        error = np.linalg.norm(transformed[0][0] - dst_pts[0])
                        print(f"   Sample error: {error:.2f} pixels")
                        
                        cv2.destroyAllWindows()
                        return H
                    else:
                        print("❌ Failed to compute homography! Try selecting better points.")
                else:
                    print(f"❌ Need at least 4 matching point pairs! (Currently: {len(self.points['src'])} source, {len(self.points['dst'])} target)")
            
            elif key == ord('r'):  # Reset points
                self.points = {'src': [], 'dst': []}
                print("\n🔄 Points reset. Start selecting again...")
            
            elif key == ord('q'):  # Skip this pair
                print("\n⏭️  Skipping this camera pair...")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def calibrate_all(self):
        """Calibrate all adjacent camera pairs"""
        print("\n" + "="*60)
        print("🎥 MULTI-CAMERA CALIBRATION WIZARD")
        print("="*60)
        print("\nThis tool will help you calibrate adjacent camera pairs.")
        print("You'll select matching points in overlapping regions between cameras.\n")
        
        calibration_results = {}
        
        # Get camera pairs from config
        for cam_id, cam_data in self.config['cameras'].items():
            for neighbor_id in cam_data.get('neighbors', []):
                # Only calibrate each pair once (avoid duplicates)
                pair_key = f"{cam_id}_to_{neighbor_id}"
                reverse_key = f"{neighbor_id}_to_{cam_id}"
                
                if reverse_key not in calibration_results:
                    print(f"\n📹 Camera Pair: {cam_id} → {neighbor_id}")
                    print(f"   Overlap Region: Look for common features visible in both views")
                    
                    H = self.calibrate_pair(cam_id, neighbor_id)
                    
                    if H is not None:
                        calibration_results[pair_key] = {
                            'source': cam_id,
                            'target': neighbor_id,
                            'homography': H.tolist(),
                            'points_used': len(self.points['src'])
                        }
                        print(f"✅ Saved calibration: {pair_key}")
                    else:
                        print(f"⏭️  Skipped: {pair_key}")
        
        # Save calibration results
        if calibration_results:
            output_path = 'camera_calibration.json'
            with open(output_path, 'w') as f:
                json.dump(calibration_results, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"✅ Calibration complete! Saved to: {output_path}")
            print(f"   Total pairs calibrated: {len(calibration_results)}")
            print(f"{'='*60}\n")
            
            print("📋 Summary:")
            for pair_key, data in calibration_results.items():
                print(f"  • {data['source']} → {data['target']}: {data['points_used']} points")
            
            print("\n🚀 Next steps:")
            print("   1. Run: python process_multicam.py")
            print("   2. The system will use these homographies to track vehicles across cameras")
            print("   3. Check the output CSV for unified global coordinates\n")
        else:
            print("\n⚠️  No calibrations were saved. Please try again.")

def main():
    """Main calibration workflow"""
    try:
        calibrator = CameraCalibrator('camera_config.json')
        calibrator.calibrate_all()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure camera_config.json exists with valid video paths!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
