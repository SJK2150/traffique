"""
Calibration Tool for Multi-Camera System
Interactive GUI to compute homography matrices for coordinate transformation
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple


class CalibrationTool:
    """
    Interactive tool to calibrate cameras by selecting corresponding points
    between camera view and global map.
    """
    
    def __init__(self, 
                 camera_image: np.ndarray,
                 map_image: np.ndarray,
                 camera_name: str = "Camera"):
        """
        Initialize calibration tool.
        
        Args:
            camera_image: Image from camera view
            map_image: Global map image (e.g., Google Earth screenshot)
            camera_name: Name of the camera being calibrated
        """
        self.camera_image = camera_image.copy()
        self.map_image = map_image.copy()
        self.camera_name = camera_name
        
        # Points storage
        self.camera_points: List[Tuple[int, int]] = []
        self.map_points: List[Tuple[int, int]] = []
        
        # Display images
        self.camera_display = camera_image.copy()
        self.map_display = map_image.copy()
        
        # State
        self.current_mode = "camera"  # "camera" or "map"
        self.homography_matrix = None
        
        # Colors
        self.point_color = (0, 255, 0)
        self.text_color = (255, 255, 255)
        self.line_color = (255, 0, 0)
        
    def mouse_callback_camera(self, event, x, y, flags, param):
        """Handle mouse events for camera image."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.camera_points) < 4:
            self.camera_points.append((x, y))
            self.update_camera_display()
            
            if len(self.camera_points) == 4:
                print(f"✓ 4 points selected on camera view. Now select corresponding points on map.")
                self.current_mode = "map"
    
    def mouse_callback_map(self, event, x, y, flags, param):
        """Handle mouse events for map image."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.map_points) < 4:
            self.map_points.append((x, y))
            self.update_map_display()
            
            if len(self.map_points) == 4:
                print(f"✓ 4 points selected on map. Computing homography...")
                self.compute_homography()
    
    def update_camera_display(self):
        """Update camera display with selected points."""
        self.camera_display = self.camera_image.copy()
        
        # Draw points
        for i, point in enumerate(self.camera_points):
            cv2.circle(self.camera_display, point, 8, self.point_color, -1)
            cv2.circle(self.camera_display, point, 10, (0, 0, 0), 2)
            cv2.putText(self.camera_display, str(i + 1), 
                       (point[0] + 15, point[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
        
        # Draw lines between points
        if len(self.camera_points) > 1:
            for i in range(len(self.camera_points) - 1):
                cv2.line(self.camera_display, 
                        self.camera_points[i], 
                        self.camera_points[i + 1], 
                        self.line_color, 2)
        
        # Draw closing line if 4 points
        if len(self.camera_points) == 4:
            cv2.line(self.camera_display, 
                    self.camera_points[3], 
                    self.camera_points[0], 
                    self.line_color, 2)
    
    def update_map_display(self):
        """Update map display with selected points."""
        self.map_display = self.map_image.copy()
        
        # Draw points
        for i, point in enumerate(self.map_points):
            cv2.circle(self.map_display, point, 8, self.point_color, -1)
            cv2.circle(self.map_display, point, 10, (0, 0, 0), 2)
            cv2.putText(self.map_display, str(i + 1), 
                       (point[0] + 15, point[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
        
        # Draw lines between points
        if len(self.map_points) > 1:
            for i in range(len(self.map_points) - 1):
                cv2.line(self.map_display, 
                        self.map_points[i], 
                        self.map_points[i + 1], 
                        self.line_color, 2)
        
        # Draw closing line if 4 points
        if len(self.map_points) == 4:
            cv2.line(self.map_display, 
                    self.map_points[3], 
                    self.map_points[0], 
                    self.line_color, 2)
    
    def compute_homography(self):
        """Compute homography matrix from selected points."""
        if len(self.camera_points) != 4 or len(self.map_points) != 4:
            print("Error: Need exactly 4 points on each image")
            return
        
        # Convert to numpy arrays
        src_points = np.array(self.camera_points, dtype=np.float32)
        dst_points = np.array(self.map_points, dtype=np.float32)
        
        # Compute homography
        self.homography_matrix, status = cv2.findHomography(src_points, dst_points)
        
        if self.homography_matrix is not None:
            print(f"✓ Homography matrix computed successfully!")
            print(f"\nHomography Matrix:")
            print(self.homography_matrix)
            
            # Test transformation
            self.visualize_transformation()
        else:
            print("✗ Failed to compute homography matrix")
    
    def visualize_transformation(self):
        """Visualize the transformation by warping camera image onto map."""
        if self.homography_matrix is None:
            return
        
        # Warp camera image
        h, w = self.map_image.shape[:2]
        warped = cv2.warpPerspective(self.camera_image, 
                                     self.homography_matrix, 
                                     (w, h))
        
        # Blend with map
        alpha = 0.5
        blended = cv2.addWeighted(self.map_image, alpha, warped, 1 - alpha, 0)
        
        cv2.imshow("Transformation Result (Blended)", blended)
    
    def reset_points(self):
        """Reset all selected points."""
        self.camera_points.clear()
        self.map_points.clear()
        self.camera_display = self.camera_image.copy()
        self.map_display = self.map_image.copy()
        self.current_mode = "camera"
        self.homography_matrix = None
        print("Points reset. Start over.")
    
    def run(self) -> np.ndarray:
        """
        Run the calibration tool.
        
        Returns:
            Computed homography matrix
        """
        print(f"\n{'='*60}")
        print(f"Camera Calibration Tool - {self.camera_name}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("1. Click 4 corresponding points on the CAMERA view (e.g., road corners)")
        print("2. Click the same 4 points on the MAP view in the same order")
        print("3. Press 's' to save the homography matrix")
        print("4. Press 'r' to reset and start over")
        print("5. Press 'q' to quit\n")
        
        # Create windows
        cv2.namedWindow(f"{self.camera_name} View")
        cv2.namedWindow("Global Map View")
        
        # Set mouse callbacks
        cv2.setMouseCallback(f"{self.camera_name} View", self.mouse_callback_camera)
        cv2.setMouseCallback("Global Map View", self.mouse_callback_map)
        
        while True:
            # Add instructions overlay
            camera_with_text = self.camera_display.copy()
            map_with_text = self.map_display.copy()
            
            # Camera view instructions
            if len(self.camera_points) < 4:
                cv2.putText(camera_with_text, 
                           f"Select point {len(self.camera_points) + 1}/4", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(camera_with_text, 
                           "4 points selected", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Map view instructions
            if len(self.camera_points) == 4 and len(self.map_points) < 4:
                cv2.putText(map_with_text, 
                           f"Select corresponding point {len(self.map_points) + 1}/4", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif len(self.map_points) == 4:
                cv2.putText(map_with_text, 
                           "4 points selected - Press 's' to save", 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show images
            cv2.imshow(f"{self.camera_name} View", camera_with_text)
            cv2.imshow("Global Map View", map_with_text)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Calibration cancelled.")
                break
            elif key == ord('r'):
                self.reset_points()
            elif key == ord('s') and self.homography_matrix is not None:
                print("✓ Homography matrix ready to save.")
                break
        
        cv2.destroyAllWindows()
        return self.homography_matrix


def calibrate_camera(camera_image_path: str,
                     map_image_path: str,
                     output_path: str,
                     camera_name: str = "Camera") -> bool:
    """
    Calibrate a camera and save homography matrix.
    
    Args:
        camera_image_path: Path to camera frame image
        map_image_path: Path to global map image
        output_path: Path to save homography matrix (.npy)
        camera_name: Name of the camera
        
    Returns:
        True if calibration successful
    """
    # Load images
    camera_img = cv2.imread(camera_image_path)
    map_img = cv2.imread(map_image_path)
    
    if camera_img is None:
        print(f"Error: Cannot load camera image: {camera_image_path}")
        return False
    
    if map_img is None:
        print(f"Error: Cannot load map image: {map_image_path}")
        return False
    
    print(f"Camera image size: {camera_img.shape[1]}x{camera_img.shape[0]}")
    print(f"Map image size: {map_img.shape[1]}x{map_img.shape[0]}")
    
    # Run calibration tool
    tool = CalibrationTool(camera_img, map_img, camera_name)
    H = tool.run()
    
    if H is not None:
        # Save homography matrix
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, H)
        print(f"✓ Homography matrix saved to: {output_path}")
        
        # Also save as text for inspection
        txt_path = output_file.with_suffix('.txt')
        np.savetxt(txt_path, H, fmt='%.8f')
        print(f"✓ Human-readable matrix saved to: {txt_path}")
        
        return True
    
    return False


def load_homography(path: str) -> np.ndarray:
    """
    Load homography matrix from file.
    
    Args:
        path: Path to .npy file
        
    Returns:
        Homography matrix
    """
    return np.load(path)


def transform_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Transform a point using homography matrix.
    
    Args:
        point: (x, y) coordinates in source image
        H: Homography matrix
        
    Returns:
        Transformed (x, y) coordinates in destination image
    """
    pt = np.array([point[0], point[1], 1.0])
    transformed = H @ pt
    transformed = transformed / transformed[2]
    return (transformed[0], transformed[1])


def batch_transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform multiple points using homography matrix.
    
    Args:
        points: Array of shape (N, 2) with (x, y) coordinates
        H: Homography matrix
        
    Returns:
        Transformed points array of shape (N, 2)
    """
    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Transform
    transformed = (H @ points_h.T).T
    
    # Normalize
    transformed = transformed / transformed[:, 2:3]
    
    return transformed[:, :2]


def main():
    parser = argparse.ArgumentParser(
        description="Calibration tool for multi-camera coordinate transformation"
    )
    
    parser.add_argument(
        "--camera-image", "-c",
        type=str,
        required=True,
        help="Path to camera frame image"
    )
    
    parser.add_argument(
        "--map-image", "-m",
        type=str,
        required=True,
        help="Path to global map image"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for homography matrix (.npy)"
    )
    
    parser.add_argument(
        "--camera-name", "-n",
        type=str,
        default="Camera",
        help="Name of the camera"
    )
    
    args = parser.parse_args()
    
    success = calibrate_camera(
        camera_image_path=args.camera_image,
        map_image_path=args.map_image,
        output_path=args.output,
        camera_name=args.camera_name
    )
    
    if success:
        print("\n✓ Calibration completed successfully!")
    else:
        print("\n✗ Calibration failed.")


if __name__ == "__main__":
    main()
