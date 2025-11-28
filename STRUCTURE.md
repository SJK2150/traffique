# Expected Directory Structure

After setup and running the system, your project should look like this:

```
iitmcvproj/
│
├── 📄 Core Python Files
│   ├── main.py                      # Main pipeline
│   ├── vehicle.py                   # Vehicle tracking classes
│   ├── fusion.py                    # Multi-camera fusion
│   ├── analytics.py                 # Analytics and logging
│   ├── calibration.py               # Calibration tool
│   ├── train_model.py               # Training script
│   ├── extract_frames.py            # Frame extraction
│   ├── analyze_results.py           # Results analysis
│   └── setup.py                     # Setup utility
│
├── 📋 Configuration Files
│   ├── config.yaml                  # Main configuration
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                   # Git ignore rules
│
├── 📚 Documentation
│   ├── README.md                    # Complete documentation
│   ├── QUICKSTART.md               # Quick start guide
│   ├── COMMANDS.md                 # Command reference
│   ├── IMPLEMENTATION_SUMMARY.md   # Project summary
│   └── STRUCTURE.md                # This file
│
├── 📂 data/
│   │
│   ├── 🎥 videos/                  # Input videos
│   │   ├── camera1.mp4             # Camera 1 footage
│   │   ├── camera2.mp4             # Camera 2 footage
│   │   ├── camera1_frame.jpg       # Extracted frame for calibration
│   │   └── camera2_frame.jpg       # Extracted frame for calibration
│   │
│   ├── 🗺️ maps/                    # Global reference maps
│   │   └── global_map.jpg          # Google Earth screenshot
│   │
│   ├── 📐 calibration/             # Homography matrices
│   │   ├── camera1_H.npy           # Camera 1 homography
│   │   ├── camera1_H.txt           # Human-readable matrix
│   │   ├── camera2_H.npy           # Camera 2 homography
│   │   └── camera2_H.txt           # Human-readable matrix
│   │
│   └── 📊 dataset/                 # Training dataset
│       ├── raw_frames/             # Extracted frames for labeling
│       │   ├── frame_000000_t0.00s.jpg
│       │   ├── frame_000001_t1.00s.jpg
│       │   └── ...
│       │
│       ├── images/                 # Organized images
│       │   ├── train/              # Training images (70%)
│       │   │   ├── img_001.jpg
│       │   │   └── ...
│       │   ├── val/                # Validation images (20%)
│       │   │   ├── img_150.jpg
│       │   │   └── ...
│       │   └── test/               # Test images (10%)
│       │       ├── img_180.jpg
│       │       └── ...
│       │
│       ├── labels/                 # YOLO format labels
│       │   ├── train/              # Training labels
│       │   │   ├── img_001.txt
│       │   │   └── ...
│       │   ├── val/                # Validation labels
│       │   │   ├── img_150.txt
│       │   │   └── ...
│       │   └── test/               # Test labels
│       │       ├── img_180.txt
│       │       └── ...
│       │
│       └── dataset.yaml            # Dataset configuration
│
├── 🤖 models/                      # Trained models
│   └── best.pt                     # Your custom trained model
│
├── 🏃 runs/                        # Training runs
│   └── train/
│       └── drone_traffic/          # Training experiment
│           ├── weights/
│           │   ├── best.pt         # Best model weights
│           │   └── last.pt         # Last epoch weights
│           ├── results.csv         # Training metrics
│           ├── confusion_matrix.png
│           ├── results.png
│           └── ...
│
├── 📤 output/                      # System outputs
│   ├── traffic_data.csv            # Vehicle tracking log
│   ├── heatmap.png                 # Traffic heatmap
│   ├── result.mp4                  # Visualization video
│   │
│   └── analysis/                   # Analysis reports
│       ├── report.html             # HTML report
│       ├── vehicle_summaries.csv   # Per-vehicle summary
│       ├── vehicle_counts_over_time.png
│       ├── class_distribution.png
│       ├── trajectory_lengths.png
│       ├── speed_distribution.png
│       ├── camera_coverage.png
│       └── spatial_heatmap.png
│
├── 💾 backups/                     # Backups (optional)
│   └── 20251127_143000/
│       ├── traffic_data.csv
│       ├── config.yaml
│       └── ...
│
└── 🐍 venv/                        # Virtual environment
    ├── Scripts/
    ├── Lib/
    └── ...
```

---

## File Descriptions

### Core Files

| File | Purpose | Size | Type |
|------|---------|------|------|
| `main.py` | Main pipeline orchestration | ~15 KB | Python |
| `vehicle.py` | Vehicle tracking system | ~12 KB | Python |
| `fusion.py` | Multi-camera fusion logic | ~15 KB | Python |
| `analytics.py` | Analytics and logging | ~16 KB | Python |
| `calibration.py` | Camera calibration tool | ~12 KB | Python |
| `train_model.py` | Model training pipeline | ~14 KB | Python |
| `extract_frames.py` | Frame extraction utility | ~8 KB | Python |
| `analyze_results.py` | Results analysis | ~14 KB | Python |

### Data Files

| File | Purpose | Format | Typical Size |
|------|---------|--------|--------------|
| `camera1.mp4` | Camera 1 video feed | Video | 100 MB - 1 GB |
| `global_map.jpg` | Reference map | Image | 500 KB - 5 MB |
| `camera1_H.npy` | Homography matrix | NumPy | 144 bytes |
| `best.pt` | Trained model | PyTorch | 6-50 MB |
| `traffic_data.csv` | Tracking log | CSV | 1-100 MB |

### Label Format (YOLO)

Each `.txt` file contains bounding boxes:
```
class_id center_x center_y width height
0 0.5234 0.3456 0.1234 0.0987
1 0.7123 0.6543 0.0876 0.0654
```

Where:
- `class_id`: 0=Car, 1=Bike, 2=Pedestrian
- Coordinates normalized (0-1)

### CSV Output Format

`traffic_data.csv`:
```csv
timestamp,frame,vehicle_id,class,global_x,global_y,camera_id,confidence,total_distance,average_speed,trajectory_length
2025-11-27 10:30:15.123,1,1,Car,450.2,320.5,1,0.87,0.0,0.0,1
2025-11-27 10:30:15.156,2,1,Car,451.8,322.1,1,0.89,2.3,2.3,2
```

---

## Storage Requirements

### Minimum Setup
- Python files: ~500 KB
- Dependencies (venv): ~2 GB
- Total: ~2.5 GB

### With Training Data
- Dataset (200 images): ~50 MB
- Labels: ~1 MB
- Total: ~2.5 GB

### Full Project
- Videos (2 cameras, 10 min): ~2 GB
- Dataset: ~50 MB
- Training runs: ~100 MB
- Models: ~50 MB
- Output: ~100 MB
- **Total: ~5 GB**

### Long-term Storage
For 1 hour of multi-camera footage:
- Input videos: ~12 GB
- Output video: ~1 GB
- CSV logs: ~50 MB
- **Total: ~13 GB**

---

## File Lifecycle

### Training Phase
```
Raw Videos → Frames → Labeled Data → Dataset → Trained Model
```

### Calibration Phase
```
Video Frame + Map → Point Selection → Homography Matrix
```

### Processing Phase
```
Videos + Model + Homography → Detection → Fusion → Output
```

### Analysis Phase
```
CSV Log → Analysis Script → Reports + Visualizations
```

---

## Important Files to Backup

### Essential (Cannot Regenerate)
1. ✅ Labeled dataset (`data/dataset/`)
2. ✅ Trained model (`models/best.pt`)
3. ✅ Homography matrices (`data/calibration/`)
4. ✅ Configuration (`config.yaml`)

### Important (Time-consuming to Regenerate)
5. Raw videos (`data/videos/`)
6. Global map (`data/maps/global_map.jpg`)
7. Training runs (`runs/train/`)

### Can Regenerate
- Output files (`output/`)
- Virtual environment (`venv/`)
- Temporary files

---

## Gitignore Recommendations

Files to exclude from version control:

```gitignore
# Data
data/videos/*.mp4
data/videos/*.avi
data/dataset/raw_frames/
data/dataset/images/
data/dataset/labels/

# Models
models/*.pt
runs/

# Output
output/
*.csv

# Environment
venv/
__pycache__/
*.pyc
```

Files to include:
- Source code (`.py`)
- Configuration (`.yaml`)
- Documentation (`.md`)
- Requirements (`requirements.txt`)
- Sample homography matrices (optional)

---

## Cleanup Commands

### Clean Output Files
```powershell
Remove-Item -Recurse -Force "output\*"
```

### Clean Training Runs
```powershell
Remove-Item -Recurse -Force "runs\*"
```

### Deep Clean (Keep Source Only)
```powershell
Remove-Item -Recurse -Force "venv", "output", "runs", "data\dataset\raw_frames"
```

### Fresh Start
```powershell
# Keep only source code and configs
Remove-Item -Recurse -Force "venv", "output", "runs", "data", "models"
python setup.py
```

---

## Validation Checklist

Before running the system, ensure these exist:

### Required Files
- [ ] `config.yaml` (configured with your paths)
- [ ] `data/videos/camera1.mp4` (or your video files)
- [ ] `data/maps/global_map.jpg` (your reference map)
- [ ] `data/calibration/camera1_H.npy` (calibrated)
- [ ] `models/best.pt` (trained model)

### Required Directories
- [ ] `data/videos/`
- [ ] `data/maps/`
- [ ] `data/calibration/`
- [ ] `models/`
- [ ] `output/`

### Optional but Recommended
- [ ] `data/dataset/` (if training)
- [ ] `runs/train/` (training history)
- [ ] `backups/` (backup important files)

---

## Quick Setup Commands

### Create All Directories
```powershell
python setup.py
```

### Manual Creation
```powershell
$dirs = @(
    "data\videos",
    "data\maps",
    "data\calibration",
    "data\dataset\raw_frames",
    "data\dataset\images\train",
    "data\dataset\images\val",
    "data\dataset\images\test",
    "data\dataset\labels\train",
    "data\dataset\labels\val",
    "data\dataset\labels\test",
    "models",
    "output",
    "runs\train"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir
}
```

---

## Size Management Tips

### Reduce Dataset Size
- Use fewer frames (100-150 is usually sufficient)
- Compress videos before storing
- Delete raw frames after labeling

### Reduce Model Size
- Use YOLOv8n (nano) instead of larger variants
- Export to ONNX for deployment

### Reduce Output Size
- Lower output video resolution
- Compress output videos
- Archive old results

---

This structure follows best practices for machine learning projects and maintains separation of concerns between code, data, models, and outputs.
