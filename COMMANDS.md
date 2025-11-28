# Helper Commands for Multi-Camera Traffic Analysis

This file contains useful PowerShell commands for common tasks.

## Setup & Installation

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Verify installation
python -c "from ultralytics import YOLO; import cv2; print('✓ All imports OK')"
```

## Data Preparation

### Extract Frames for Labeling

```powershell
# Extract 150 frames at 1 fps from single video
python extract_frames.py --video "data\videos\drone1.mp4" --output "data\dataset\raw_frames" --fps 1 --max-frames 150

# Extract from multiple videos
python extract_frames.py --videos "data\videos\cam1.mp4" "data\videos\cam2.mp4" --output "data\dataset\raw_frames" --fps 1 --frames-per-video 150

# Create dataset structure
python extract_frames.py --create-structure --dataset-dir "data\dataset"
```

### Extract Single Frame for Calibration

```powershell
# Using ffmpeg
ffmpeg -i "data\videos\camera1.mp4" -vframes 1 "data\videos\camera1_frame.jpg"

# Using Python
python -c "import cv2; cap=cv2.VideoCapture('data/videos/camera1.mp4'); ret,frame=cap.read(); cv2.imwrite('data/videos/camera1_frame.jpg',frame); cap.release(); print('✓ Frame extracted')"
```

## Model Training

### Prepare Dataset

```powershell
# Prepare and split dataset
python train_model.py prepare `
  --dataset-dir "data\dataset" `
  --source-images "data\labeled\images" `
  --source-labels "data\labeled\labels" `
  --split
```

### Train Model

```powershell
# Quick training (20 epochs for testing)
python train_model.py train `
  --dataset "data\dataset\dataset.yaml" `
  --model yolov8n.pt `
  --epochs 20 `
  --batch 16 `
  --device 0

# Full training (50 epochs)
python train_model.py train `
  --dataset "data\dataset\dataset.yaml" `
  --model yolov8n.pt `
  --epochs 50 `
  --batch 16 `
  --imgsz 640 `
  --device 0 `
  --project "runs\train" `
  --name "drone_traffic"

# High-quality training (100 epochs, larger batch)
python train_model.py train `
  --dataset "data\dataset\dataset.yaml" `
  --model yolov8s.pt `
  --epochs 100 `
  --batch 32 `
  --imgsz 640 `
  --device 0
```

### Validate Model

```powershell
# Validate on test set
python train_model.py validate `
  --model "runs\train\drone_traffic\weights\best.pt" `
  --dataset "data\dataset\dataset.yaml" `
  --device 0
```

### Test Inference

```powershell
# Test on single image
python train_model.py test `
  --model "runs\train\drone_traffic\weights\best.pt" `
  --image "data\test_images\test.jpg" `
  --output "output\test" `
  --conf 0.25
```

### Export Model

```powershell
# Export to ONNX
python train_model.py export `
  --model "runs\train\drone_traffic\weights\best.pt" `
  --format onnx

# Export to TensorRT (requires TensorRT installed)
python train_model.py export `
  --model "runs\train\drone_traffic\weights\best.pt" `
  --format engine
```

## Camera Calibration

```powershell
# Calibrate Camera 1
python calibration.py `
  --camera-image "data\videos\camera1_frame.jpg" `
  --map-image "data\maps\global_map.jpg" `
  --output "data\calibration\camera1_H.npy" `
  --camera-name "Camera 1"

# Calibrate Camera 2
python calibration.py `
  --camera-image "data\videos\camera2_frame.jpg" `
  --map-image "data\maps\global_map.jpg" `
  --output "data\calibration\camera2_H.npy" `
  --camera-name "Camera 2"

# Calibrate Camera 3
python calibration.py `
  --camera-image "data\videos\camera3_frame.jpg" `
  --map-image "data\maps\global_map.jpg" `
  --output "data\calibration\camera3_H.npy" `
  --camera-name "Camera 3"
```

## Run System

### Basic Run

```powershell
# Run with default config
python main.py --config config.yaml

# Run with interactive zone setup
python main.py --config config.yaml --setup-zones
```

### Custom Configuration

```powershell
# Run with custom config file
python main.py --config "configs\custom_config.yaml"
```

## Analysis & Reporting

### Generate Analysis Report

```powershell
# Full analysis with all plots
python analyze_results.py `
  --csv "output\traffic_data.csv" `
  --output "output\analysis"

# Summary statistics only
python analyze_results.py `
  --csv "output\traffic_data.csv" `
  --summary
```

### View Results

```powershell
# Open HTML report in browser
start "output\analysis\report.html"

# Open output video
start "output\result.mp4"

# View CSV in Excel
start excel "output\traffic_data.csv"
```

## Video Processing

### Convert Video Format

```powershell
# Convert to MP4
ffmpeg -i "input.avi" -c:v libx264 -crf 23 "output.mp4"

# Reduce file size
ffmpeg -i "input.mp4" -c:v libx264 -crf 28 "output_compressed.mp4"
```

### Extract Segment

```powershell
# Extract first 1 minute
ffmpeg -i "input.mp4" -t 60 -c copy "output_1min.mp4"

# Extract from 1:30 to 2:30
ffmpeg -i "input.mp4" -ss 00:01:30 -to 00:02:30 -c copy "output_segment.mp4"
```

### Get Video Info

```powershell
# Video properties
ffprobe -v error -show_entries format=duration,size,bit_rate -show_entries stream=width,height,r_frame_rate "video.mp4"
```

## Debugging

### Check GPU

```powershell
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"
```

### Test Model

```powershell
# Quick inference test
python -c "from ultralytics import YOLO; model = YOLO('runs/train/drone_traffic/weights/best.pt'); results = model('data/test.jpg'); print('✓ Model working')"
```

### Check Calibration

```powershell
# Load and verify homography matrix
python -c "import numpy as np; H = np.load('data/calibration/camera1_H.npy'); print('Homography Matrix:'); print(H)"
```

### Validate Config

```powershell
# Check config syntax
python -c "import yaml; config = yaml.safe_load(open('config.yaml')); print('✓ Config valid'); print(f'Cameras: {len(config[\"cameras\"])}')"
```

## File Management

### Clean Output

```powershell
# Remove all output files
Remove-Item -Recurse -Force "output\*"

# Remove training runs
Remove-Item -Recurse -Force "runs\*"
```

### Backup Results

```powershell
# Create backup with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path "backups\$timestamp"
Copy-Item -Recurse "output\*" "backups\$timestamp\"
Copy-Item "config.yaml" "backups\$timestamp\"
```

### Archive Dataset

```powershell
# Compress dataset
Compress-Archive -Path "data\dataset\*" -DestinationPath "dataset_backup.zip"
```

## Monitoring

### Watch Training Progress

```powershell
# View training results in real-time
Get-Content "runs\train\drone_traffic\results.csv" -Wait -Tail 10
```

### Monitor System Resources

```powershell
# Check GPU usage (requires nvidia-smi)
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# Check CPU and RAM
Get-Process python | Format-Table ProcessName, CPU, WorkingSet -AutoSize
```

## Batch Processing

### Process Multiple Videos

```powershell
# Loop through all videos in directory
Get-ChildItem "data\videos\*.mp4" | ForEach-Object {
    Write-Host "Processing: $($_.Name)"
    python main.py --config "config.yaml"
}
```

### Train Multiple Configurations

```powershell
# Train with different epochs
foreach ($epochs in @(20, 50, 100)) {
    python train_model.py train `
      --dataset "data\dataset\dataset.yaml" `
      --epochs $epochs `
      --name "drone_traffic_$($epochs)ep"
}
```

## Tips & Tricks

### Quick Test on Short Clip

```powershell
# Extract short clip for testing
ffmpeg -i "data\videos\camera1.mp4" -t 30 -c copy "data\videos\camera1_test.mp4"

# Update config to use test clip
# Then run normally
python main.py
```

### Resume Training

```powershell
# Resume from last checkpoint
python train_model.py train `
  --dataset "data\dataset\dataset.yaml" `
  --model "runs\train\drone_traffic\weights\last.pt" `
  --epochs 100
```

### Export Specific Frame

```powershell
# Export frame at specific time (e.g., 1:30)
ffmpeg -ss 00:01:30 -i "data\videos\camera1.mp4" -vframes 1 "camera1_t90s.jpg"
```

## Useful Python One-liners

```powershell
# Count labeled images
python -c "from pathlib import Path; print(f'Images: {len(list(Path(\"data/dataset/images/train\").glob(\"*.jpg\")))}')"

# Check model size
python -c "from pathlib import Path; size = Path('runs/train/drone_traffic/weights/best.pt').stat().st_size / 1024 / 1024; print(f'Model size: {size:.1f} MB')"

# Count vehicles in CSV
python -c "import pandas as pd; df = pd.read_csv('output/traffic_data.csv'); print(f'Unique vehicles: {df[\"vehicle_id\"].nunique()}')"
```

## Troubleshooting Commands

```powershell
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Clear Python cache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Check disk space
Get-PSDrive C

# Find large files
Get-ChildItem -Recurse | Where-Object { $_.Length -gt 100MB } | Sort-Object Length -Descending | Select-Object Name, @{Name="Size(MB)";Expression={$_.Length / 1MB}}
```

## Performance Profiling

```powershell
# Time execution
Measure-Command { python main.py --config config.yaml }

# Profile with verbose output
python -m cProfile -o output.prof main.py --config config.yaml

# View profile
python -m pstats output.prof
```

---

**Note**: Replace paths and filenames with your actual file locations.

For more information, see README.md and QUICKSTART.md.
