# Project Structure & File Functions Guide

## 📍 Frontend & Backend Locations

### **Frontend (Web UI)**
- **Location:** `frontend/` directory
- **Main files:**
  - `frontend/index.html` - Main UI
  - `frontend/src/` - React/Vue components
  - `frontend/vite.config.js` - Build configuration
  - `frontend/tailwind.config.js` - Styling

### **Backend (API Server)**
- **Main file:** `api_server.py` - REST API server
- **Port:** Default 5000
- **Supports:** Video upload, analysis, results retrieval

---

## 🎬 Core Analysis Pipeline

### **Main Entry Point**
- **`scripts/run_d1f1_improved.py`** (LATEST)
  - Purpose: End-to-end video analysis with Kalman-filtered tracking
  - Features: VisDrone detection, SAHI slicing, Kalman tracking, trajectory smoothing
  - Usage: `python scripts/run_d1f1_improved.py --video D1F1_stab_cropped.mp4 --frame 9861 --time_window 15 --use_sahi`
  - Output: `output/*_improved_tracks.csv`, `output/*_improved_trajectories.jpg`

### **Visualization**
- **`scripts/show_top5_improved.py`** (LATEST)
  - Purpose: Visualize top 5 vehicles with their trajectories
  - Usage: `python scripts/show_top5_improved.py`
  - Output: `output/D1F1_stab_top5_improved_trajectories.jpg`

---

## 🚗 Tracking & Detection

### **Improved Tracker (LATEST)**
- **`utils/improved_tracker.py`**
  - Contains: KalmanFilterTrack, KalmanTrack, ImprovedOnlineTracker classes
  - Features: Per-coordinate Kalman filtering, motion prediction, appearance-motion fusion
  - Replaces: Old greedy tracker (deleted)
  - Uses: Velocity model, covariance estimation

### **Re-identification (ReID)**
- **`utils/reid.py`**
  - Purpose: Extract vehicle appearance embeddings
  - Model: OSNet (lightweight appearance model)
  - Used by: ImprovedOnlineTracker for matching vehicles across frames

### **Trajectory Utilities**
- **`utils/trajectory.py`**
  - Functions: Kalman smoothing, linear interpolation, gap filling
  - Used by: run_d1f1_improved.py for trajectory post-processing
  - Features: Smooth noisy trajectories, fill missing frames

---

## 🎥 Video Processing

### **Stabilization & Cropping (LATEST)**
- **`scripts/high_quality_stabilize.py`**
  - Purpose: High-quality video stabilization using KLT feature tracking
  - Output: Transform matrices + crop box JSON
  - Method: Affine estimation → Gaussian smoothing
  - Usage: `python scripts/high_quality_stabilize.py --video D1F1_stab.mp4 --frame 9861 --time_window 15`

### **Crop Computation (FALLBACK)**
- **`scripts/compute_constant_crop.py`**
  - Purpose: Compute safe crop box from stabilization transforms
  - Usage: When you already have transform matrices
  - Output: `output/constant_crop.json`

---

## 📊 Analytics & Visualization

### **Interactive Analytics Engine**
- **`interactive_analytics.py`**
  - Purpose: Main analytics computation engine
  - Contains: VehicleAnalyzer class for trajectory analysis
  - Used by: run_d1f1_improved.py, api_server.py
  - Features: Trajectory extraction, vehicle metrics, speed calculation

### **Vehicle Data Structures**
- **`vehicle.py`**
  - Purpose: Vehicle class definitions and data models
  - Used by: API server, analytics engine
  - Contains: Vehicle, Track, BoundingBox classes

---

## ⚙️ Utilities & Configuration

### **Setup & Initialization**
- **`setup.py`** - Package installation (if distributing)
- **`utils/__init__.py`** - Package initialization

### **Configuration Files**
- **`config.yaml`** - Global configuration
- **`camera_config.json`** - Camera parameters
- **`camera_calibration.json`** - Calibration data
- **`output/constant_crop.json`** - Current crop box

---

## 📁 Project Directory Structure

```
iitmcvproj/
├── frontend/                          # Web UI (DO NOT MODIFY)
│   ├── index.html
│   ├── src/
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── scripts/                           # Analysis pipelines
│   ├── run_d1f1_improved.py          # ⭐ MAIN: Improved analysis pipeline
│   ├── show_top5_improved.py         # ⭐ LATEST: Trajectory visualization
│   ├── high_quality_stabilize.py     # ⭐ LATEST: Stabilization + crop
│   └── compute_constant_crop.py      # Fallback: Crop computation
│
├── utils/                             # Core modules
│   ├── improved_tracker.py            # ⭐ LATEST: Kalman-based tracker
│   ├── trajectory.py                  # Trajectory smoothing utilities
│   ├── reid.py                        # Vehicle appearance embeddings
│   └── __init__.py
│
├── models/                            # Pre-trained models directory
│
├── output/                            # Results & artifacts
│   ├── D1F1_stab_cropped.mp4         # Processed video
│   ├── D1F1_stab_cropped_improved_tracks.csv  # Track results
│   ├── D1F1_stab_cropped_improved_trajectories.jpg
│   ├── D1F1_stab_top5_improved_trajectories.jpg
│   └── constant_crop.json            # Crop configuration
│
├── api_server.py                      # ⭐ REST API backend
├── interactive_analytics.py           # Analytics computation engine
├── vehicle.py                         # Data structures
├── main.py                            # Entry point
├── calibration.py                     # Camera calibration utilities
├── config.yaml                        # Configuration
└── requirements.txt                   # Dependencies
```

---

## 🔄 Data Flow

```
Video Input
    ↓
high_quality_stabilize.py (stabilization + crop)
    ↓
run_d1f1_improved.py (detection + tracking)
    │
    ├→ VisDrone model (detection)
    ├→ SAHI (slicing for small objects)
    ├→ reid.py (appearance extraction)
    ├→ improved_tracker.py (Kalman tracking)
    └→ trajectory.py (smoothing)
    ↓
output/*_improved_tracks.csv
    ↓
show_top5_improved.py (visualization)
    ↓
output/*_trajectories.jpg
```

---

## 🚀 Quick Start

```bash
# 1. Stabilize video
python scripts/high_quality_stabilize.py --video D1F1_stab.mp4 --frame 9861 --time_window 15

# 2. Run improved analysis
python scripts/run_d1f1_improved.py --video D1F1_stab_cropped.mp4 --frame 9861 --time_window 15

# 3. Visualize results
python scripts/show_top5_improved.py
```

---

## 📌 Key Improvements (Latest)

✅ **Kalman-filtered tracking** - Better motion prediction, fewer ID switches
✅ **VisDrone + SAHI** - Detects 484 vehicles (vs 6 with old method)
✅ **Trajectory smoothing** - Removes detector jitter
✅ **GPU acceleration** - RTX 3050 support
✅ **620+ stable tracks** - Continuous tracking across 15-second clips

---

## 🗑️ Deleted (Obsolete)

- Old tracker: `utils/onlinetracker.py`
- Old pipeline: `scripts/run_d1f1_analysis.py`
- Old visualizations: `show_top5_trajectories.py`, `show_top5_vehicles.py`
- Testing files: Kept (can use for verification)
