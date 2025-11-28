# Quick Start Guide

This guide will get you up and running quickly.

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed (optional)
- [ ] CUDA installed (optional, for GPU acceleration)
- [ ] Drone footage videos ready
- [ ] Google Earth screenshot of the area

## Installation (5 minutes)

### 1. Setup Environment

```powershell
# Navigate to project
cd c:\Users\sakth\Documents\Projects\iitmcvproj

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Verify Installation

```powershell
# Check YOLO installation
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Check OpenCV
python -c "import cv2; print('OpenCV OK')"
```

## Quick Test (Optional)

### Download Sample Data

If you want to test with sample data first:

1. Download sample videos and place in `data/videos/`
2. Download sample map and place in `data/maps/global_map.jpg`
3. Use pre-trained YOLOv8n model (downloads automatically)

### Run with Pretrained Model

```powershell
# Update config.yaml to use base model
# Change: model.path: "yolov8n.pt"

python main.py
```

## Your First Complete Run

### Minimal Working Example (1-2 hours)

#### 1. Extract Frames (5 min)

```powershell
python extract_frames.py --video "data\videos\your_video.mp4" --output "data\dataset\raw_frames" --fps 1 --max-frames 100
```

#### 2. Label Data (30-45 min)

1. Go to [Roboflow](https://roboflow.com)
2. Create project, upload 100 frames
3. Label: Car, Bike, Pedestrian
4. Export as YOLOv8 format
5. Extract to `data/dataset/`

#### 3. Train Model (15-30 min)

```powershell
# Prepare dataset
python train_model.py prepare --dataset-dir "data\dataset" --source-images "data\dataset\train\images" --source-labels "data\dataset\train\labels" --split

# Quick training (fewer epochs for testing)
python train_model.py train --dataset "data\dataset\dataset.yaml" --epochs 20 --batch 16

# Model saved to: runs/train/drone_traffic/weights/best.pt
```

#### 4. Calibrate Cameras (10 min)

```powershell
# Extract frame for calibration
python -c "import cv2; cap=cv2.VideoCapture('data/videos/camera1.mp4'); ret,frame=cap.read(); cv2.imwrite('data/videos/cam1_frame.jpg',frame)"

# Calibrate
python calibration.py --camera-image "data\videos\cam1_frame.jpg" --map-image "data\maps\global_map.jpg" --output "data\calibration\camera1_H.npy" --camera-name "Camera 1"

# Repeat for camera 2
```

#### 5. Configure (2 min)

Edit `config.yaml`:

```yaml
model:
  path: "runs/train/drone_traffic/weights/best.pt"

cameras:
  - id: 1
    video_path: "data/videos/camera1.mp4"
    homography_matrix: "data/calibration/camera1_H.npy"
    enabled: true
    
  - id: 2
    video_path: "data/videos/camera2.mp4"
    homography_matrix: "data/calibration/camera2_H.npy"
    enabled: true

global_map:
  image_path: "data/maps/global_map.jpg"
```

#### 6. Run System (5 min)

```powershell
python main.py --config config.yaml --setup-zones
```

**Controls**:
- Press `a` to add zones
- Click to define polygon
- Press `s` to save
- Press `q` to start processing

## Expected Results

After running, you should see:

1. **Real-time visualization**: Global map with vehicle trajectories
2. **CSV log**: `output/traffic_data.csv` with all vehicle positions
3. **Heatmap**: `output/heatmap.png` showing traffic density
4. **Video**: `output/result.mp4` with visualization

## Common First-Time Issues

### Issue 1: "Cannot load video"

**Solution**: Use absolute paths in config.yaml

```yaml
video_path: "C:\\Users\\sakth\\Documents\\Projects\\iitmcvproj\\data\\videos\\camera1.mp4"
```

### Issue 2: Low detection accuracy

**Solution**: Train longer (50-100 epochs) with more data (200+ images)

### Issue 3: No vehicles detected

**Solution**: Check if your classes match the model

```yaml
classes:
  0: "Car"      # Must match your training labels
  1: "Bike"
  2: "Pedestrian"
```

### Issue 4: Calibration points don't align

**Solution**: 
- Choose more distinctive points
- Ensure same order for camera and map
- Use road intersections or clear landmarks

## Performance Tuning

### Fast Mode (for testing)

```yaml
performance:
  frame_skip: 2          # Process every 3rd frame
  resize_width: 960      # Lower resolution
  
model:
  conf_threshold: 0.3    # Higher threshold = fewer detections
```

### Accuracy Mode (for final results)

```yaml
performance:
  frame_skip: 0          # Process all frames
  resize_width: 1920     # Full resolution
  
model:
  conf_threshold: 0.15   # Lower threshold = more detections
```

## Validation Checklist

Before running full analysis:

- [ ] Model achieves >0.6 mAP50 on validation set
- [ ] Calibration visually aligns with map
- [ ] Test run completes without errors
- [ ] CSV output contains expected data
- [ ] Visualization looks reasonable

## Getting Help

### Debug Information

```powershell
# Check system info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Validate model
python train_model.py validate --model "runs\train\drone_traffic\weights\best.pt" --dataset "data\dataset\dataset.yaml"

# Test single image
python train_model.py test --model "runs\train\drone_traffic\weights\best.pt" --image "data\test.jpg" --output "output\test"
```

### Log Files

Check these if issues occur:
- `runs/train/drone_traffic/results.csv` - Training metrics
- `output/traffic_data.csv` - Tracking results
- Console output for errors

## Next Steps

Once basic system works:

1. **Improve Model**: Add more labeled data, train longer
2. **Fine-tune Fusion**: Adjust `distance_threshold` based on your map scale
3. **Add More Cameras**: Expand to 3+ cameras
4. **Optimize Performance**: Adjust frame skip and resolution
5. **Custom Analytics**: Add custom zones for specific areas

## Time Estimate

| Task | Time (First Time) | Time (Subsequent) |
|------|-------------------|-------------------|
| Setup | 10 min | 2 min |
| Extract Frames | 5 min | 5 min |
| Labeling | 45 min | 30 min |
| Training | 30 min | 20 min |
| Calibration | 15 min | 5 min |
| Run System | 10 min | 5 min |
| **Total** | **~2 hours** | **~1 hour** |

Processing time depends on video length:
- 10 min video: ~5-10 min processing
- 1 hour video: ~30-60 min processing

## Success Criteria

Your system is working well if:

1. ✓ Vehicles maintain same ID across camera transitions
2. ✓ Trajectories are smooth and continuous
3. ✓ Zone counts match visual observation
4. ✓ CSV log contains all vehicle movements
5. ✓ Heatmap shows expected traffic patterns

---

**Ready to start?** Run `python setup.py` and follow the steps above!

For detailed documentation, see `README.md`.
