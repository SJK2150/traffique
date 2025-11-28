# Multi-Camera Traffic Analysis System

A distributed traffic analysis system that provides unified tracking of vehicles across multiple static drone cameras using coordinate transformation and fusion.

## 🎯 Features

- **High-Precision Detection**: YOLOv8-Nano fine-tuned on drone footage for accurate detection of cars, bikes, and pedestrians
- **Global Coordinate Fusion**: Mathematical transformation of coordinates using homography matrices (no pixel-level video stitching)
- **Unified Trajectory Tracking**: Maintains unique IDs and complete trajectory history across multiple cameras
- **Real-time Analytics**:
  - CSV logging of all vehicle movements
  - Zone-based vehicle counting
  - Distance measurement in real-world units
  - Heatmap generation
- **Interactive Tools**: Calibration and zone setup with visual interfaces

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training and inference)
- Windows/Linux/macOS

### Setup

1. **Clone or navigate to the project directory**:
```powershell
cd c:\Users\sakth\Documents\Projects\iitmcvproj
```

2. **Create virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Create necessary directories**:
```powershell
mkdir data\videos, data\maps, data\calibration, data\dataset, models, output
```

## 📁 Project Structure

```
iitmcvproj/
├── main.py                 # Main pipeline
├── train_model.py          # Model training script
├── extract_frames.py       # Frame extraction for labeling
├── calibration.py          # Camera calibration tool
├── vehicle.py              # Vehicle tracking classes
├── fusion.py               # Multi-camera fusion logic
├── analytics.py            # Analytics and logging
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── videos/            # Input videos
│   │   ├── camera1.mp4
│   │   └── camera2.mp4
│   ├── maps/              # Global reference map
│   │   └── global_map.jpg
│   ├── calibration/       # Homography matrices
│   │   ├── camera1_H.npy
│   │   └── camera2_H.npy
│   └── dataset/           # Training dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   └── best.pt            # Trained YOLOv8 model
│
└── output/
    ├── traffic_data.csv   # Vehicle tracking log
    ├── heatmap.png        # Traffic heatmap
    └── result.mp4         # Visualization video
```

## 🎬 Quick Start

### Minimal Example

If you already have trained model and calibrated cameras:

```powershell
# 1. Edit config.yaml with your paths
# 2. Run the system
python main.py --config config.yaml
```

## 📖 Complete Workflow

### Phase 1: Data Preparation

#### Step 1: Extract Frames from Drone Videos

Extract frames for labeling (1 frame per second):

```powershell
# Single video
python extract_frames.py --video "data\raw_videos\drone_footage.mp4" --output "data\dataset\raw_frames" --fps 1 --max-frames 150

# Multiple videos
python extract_frames.py --videos "data\raw_videos\cam1.mp4" "data\raw_videos\cam2.mp4" --output "data\dataset\raw_frames" --fps 1 --frames-per-video 150

# Create dataset structure
python extract_frames.py --create-structure --dataset-dir "data\dataset"
```

#### Step 2: Label Images

1. **Upload frames to labeling platform** (Roboflow or CVAT):
   - Upload images from `data/dataset/raw_frames/`
   - Create classes: `Car`, `Bike`, `Pedestrian`
   - Draw bounding boxes around all objects
   - Export in **YOLOv8 format**

2. **Organize labeled data**:
   - Place images in `data/dataset/images/`
   - Place labels in `data/dataset/labels/`

#### Step 3: Prepare Training Dataset

```powershell
# Prepare dataset and split into train/val/test
python train_model.py prepare --dataset-dir "data\dataset" --source-images "data\dataset\labeled\images" --source-labels "data\dataset\labeled\labels" --split
```

This creates:
- 70% training set
- 20% validation set
- 10% test set
- `dataset.yaml` configuration file

### Phase 2: Model Training

#### Train YOLOv8-Nano

```powershell
# Basic training (50 epochs)
python train_model.py train --dataset "data\dataset\dataset.yaml" --model yolov8n.pt --epochs 50 --batch 16 --device 0

# Advanced training with custom parameters
python train_model.py train --dataset "data\dataset\dataset.yaml" --model yolov8n.pt --epochs 100 --batch 32 --imgsz 640 --device 0 --project "runs\train" --name "drone_traffic_v1"
```

Training outputs:
- Best model: `runs/train/drone_traffic/weights/best.pt`
- Training plots: `runs/train/drone_traffic/`
- Logs: `runs/train/drone_traffic/results.csv`

#### Validate Model

```powershell
python train_model.py validate --model "runs\train\drone_traffic\weights\best.pt" --dataset "data\dataset\dataset.yaml"
```

#### Test Inference

```powershell
python train_model.py test --model "runs\train\drone_traffic\weights\best.pt" --image "data\test_images\test1.jpg" --output "output\test" --conf 0.25
```

#### Export Model (Optional)

```powershell
# Export to ONNX for deployment
python train_model.py export --model "runs\train\drone_traffic\weights\best.pt" --format onnx
```

### Phase 3: Camera Calibration

#### Step 1: Get Global Reference Map

1. Open Google Earth
2. Navigate to your area
3. Take screenshot showing all camera coverage areas
4. Save as `data/maps/global_map.jpg`

#### Step 2: Calibrate Each Camera

```powershell
# Camera 1
python calibration.py --camera-image "data\videos\camera1_frame.jpg" --map-image "data\maps\global_map.jpg" --output "data\calibration\camera1_H.npy" --camera-name "Camera 1"

# Camera 2
python calibration.py --camera-image "data\videos\camera2_frame.jpg" --map-image "data\maps\global_map.jpg" --output "data\calibration\camera2_H.npy" --camera-name "Camera 2"
```

**Interactive Process**:
1. Window opens with camera view
2. Click **4 corresponding points** (e.g., road corners, intersections)
3. Same 4 points on global map in **same order**
4. Press `s` to save, `r` to reset, `q` to quit

**Tips for Point Selection**:
- Choose points at road intersections or lane markings
- Spread points across the entire frame
- Select stationary, easily identifiable landmarks
- Maintain same order for both views

#### Extract Frame for Calibration

```powershell
# Extract first frame from video
ffmpeg -i "data\videos\camera1.mp4" -vframes 1 "data\videos\camera1_frame.jpg"
```

Or use OpenCV:
```python
import cv2
cap = cv2.VideoCapture("data/videos/camera1.mp4")
ret, frame = cap.read()
cv2.imwrite("data/videos/camera1_frame.jpg", frame)
cap.release()
```

### Phase 4: Configuration

Edit `config.yaml`:

```yaml
# Update model path
model:
  path: "runs/train/drone_traffic/weights/best.pt"

# Update camera configurations
cameras:
  - id: 1
    name: "Camera 1"
    video_path: "data/videos/camera1.mp4"
    homography_matrix: "data/calibration/camera1_H.npy"
    enabled: true
    
  - id: 2
    name: "Camera 2"
    video_path: "data/videos/camera2.mp4"
    homography_matrix: "data/calibration/camera2_H.npy"
    enabled: true

# Update global map
global_map:
  image_path: "data/maps/global_map.jpg"
  scale: 1.0  # Adjust based on your map scale

# Adjust fusion parameters
fusion:
  distance_threshold: 2.0  # meters
  
# Configure output
analytics:
  csv_output: "output/traffic_data.csv"
  
output:
  save_visualization: true
  output_video: "output/result.mp4"
```

### Phase 5: Run the System

#### Basic Run

```powershell
python main.py --config config.yaml
```

#### With Interactive Zone Setup

```powershell
python main.py --config config.yaml --setup-zones
```

**Interactive Controls**:
- `q`: Quit
- `p`: Pause
- `a`: Add zone (when in setup mode)
- `s`: Skip zone setup

#### Zone Setup Process:
1. Press `a` when prompted
2. Enter zone name
3. Click points to define polygon on global map
4. Press `s` to save zone
5. Repeat for more zones or press `q` to finish

## ⚙️ Configuration

### Key Configuration Parameters

#### Model Settings
```yaml
model:
  conf_threshold: 0.25  # Detection confidence threshold
  iou_threshold: 0.45   # NMS IoU threshold
```

#### Fusion Settings
```yaml
fusion:
  distance_threshold: 2.0   # Max distance (meters) to merge vehicles
  frame_window: 5           # Frames to look back
  confidence_weight: 0.3    # Weight of confidence in fusion
```

#### Tracking Settings
```yaml
tracking:
  track_buffer: 30          # Frames to keep lost tracks
  match_threshold: 0.8      # ByteTrack matching threshold
```

#### Performance Settings
```yaml
performance:
  frame_skip: 0            # Process every Nth frame (0 = no skip)
  resize_width: 1280       # Resize for faster processing
  use_gpu: true
```

## 💡 Usage Examples

### Example 1: Basic Traffic Monitoring

```powershell
# Run with default settings
python main.py
```

### Example 2: High-Accuracy Mode

Edit `config.yaml`:
```yaml
model:
  conf_threshold: 0.35  # Higher threshold
  
performance:
  frame_skip: 0  # Process every frame
  resize_width: 1920  # Full resolution
```

```powershell
python main.py
```

### Example 3: Fast Processing Mode

```yaml
performance:
  frame_skip: 2  # Process every 3rd frame
  resize_width: 960  # Lower resolution
```

### Example 4: Specific Camera Analysis

```yaml
cameras:
  - id: 1
    enabled: true  # Enable Camera 1
    
  - id: 2
    enabled: false  # Disable Camera 2
```

## 🏗️ Architecture

### System Components

1. **CameraProcessor**: Handles video input and object detection per camera
2. **MultiCameraFusion**: Transforms coordinates and merges detections
3. **VehicleManager**: Maintains vehicle states and trajectories
4. **TrafficAnalytics**: Provides counting, logging, and visualization
5. **CoordinateFusion**: Implements fusion algorithm

### Data Flow

```
Camera 1 Video ──→ YOLO Detection ──→ Local Tracking ──┐
                                                         │
Camera 2 Video ──→ YOLO Detection ──→ Local Tracking ──┤
                                                         │
                                                         ↓
                                           Coordinate Transformation
                                                    (Homography)
                                                         │
                                                         ↓
                                              Global Coordinate Fusion
                                                         │
                                                         ↓
                                             Unified Vehicle Tracking
                                                         │
                                    ┌────────────────────┼────────────────────┐
                                    ↓                    ↓                    ↓
                              CSV Logging        Zone Counting        Visualization
```

### Coordinate Transformation

Each camera view point is transformed to global coordinates:

```
Point (x, y) in Camera → Homography Matrix H → Point (X, Y) in Global Map
```

Formula:
```
[X]   [h11 h12 h13]   [x]
[Y] = [h21 h22 h23] × [y]
[1]   [h31 h32 h33]   [1]
```

### Fusion Algorithm

1. **Spatial Matching**: Compare global coordinates
2. **Temporal Consistency**: Use velocity prediction
3. **Confidence Weighting**: Factor in detection confidence
4. **Cross-Camera Tracking**: Handle camera transitions

## 📊 Output Files

### CSV Log Format

`output/traffic_data.csv`:
```csv
timestamp,frame,vehicle_id,class,global_x,global_y,camera_id,confidence,total_distance,average_speed,trajectory_length
2025-11-27 10:30:15.123,1,1,Car,450.2,320.5,1,0.87,0.0,0.0,1
2025-11-27 10:30:15.156,2,1,Car,451.8,322.1,1,0.89,2.3,2.3,2
```

### Heatmap

Visual representation of traffic density saved as `output/heatmap.png`.

### Visualization Video

Real-time tracking visualization saved as `output/result.mp4` showing:
- Vehicle trajectories on global map
- Current positions with IDs
- Zone overlays
- Statistics overlay

## 🔧 Troubleshooting

### Common Issues

#### 1. "Cannot load video"
```powershell
# Check video path and codec
ffmpeg -i "data\videos\camera1.mp4"
```

#### 2. "No homography matrix"
- Ensure calibration completed successfully
- Check file paths in config.yaml
- Verify .npy files exist

#### 3. Low detection accuracy
- Increase confidence threshold
- Retrain model with more data
- Check lighting conditions match training data

#### 4. Poor fusion (duplicate IDs)
- Adjust `distance_threshold` in config
- Improve calibration accuracy
- Check homography matrix quality

#### 5. Slow performance
```yaml
performance:
  frame_skip: 1        # Process every other frame
  resize_width: 960    # Reduce resolution
```

#### 6. Out of memory
- Reduce batch size during training
- Lower `resize_width` during inference
- Process fewer cameras simultaneously

### Debug Mode

Add verbose logging:
```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed
2. **Batch Processing**: Process multiple videos separately then combine logs
3. **Resolution**: Balance between accuracy and speed
4. **Frame Skipping**: For long videos, process every 2-3 frames
5. **Model Optimization**: Export to ONNX or TensorRT for faster inference

## 🤝 Contributing

This is a project-specific implementation. For improvements:
1. Test thoroughly with your drone footage
2. Document parameter changes
3. Validate results against ground truth

## 📝 Citation

If you use this system in research:

```bibtex
@software{multi_camera_traffic_analysis,
  title = {Multi-Camera Traffic Analysis System},
  author = {Your Name},
  year = {2025},
  description = {Coordinate-based fusion for multi-camera traffic tracking}
}
```

## 📄 License

Specify your license here.

## 🙏 Acknowledgments

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- ByteTrack: https://github.com/ifzhang/ByteTrack
- OpenCV: https://opencv.org/

## 📧 Contact

For questions or issues, please refer to your project documentation or contact your supervisor.

---

**Note**: This system is designed for academic/research purposes. Ensure you have proper permissions for video data collection and processing.
