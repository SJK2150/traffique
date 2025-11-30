# Traffic Vehicle Analytics System

AI-powered vehicle detection and tracking system with interactive web interface. Uses YOLOv8-VisDrone model with SAHI for accurate vehicle detection and trajectory analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (recommended)

### Backend Setup

1. **Create Python environment**
```bash
conda create -n iitmlab python=3.10
conda activate iitmlab
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the API server**
```bash
python api_server.py
```
Backend runs at `http://localhost:5000`

### Frontend Setup

1. **Install packages**
```bash
cd frontend
npm install
```

2. **Start development server**
```bash
npm run dev
```
Frontend runs at `http://localhost:5174`

## 📖 Usage

1. Open `http://localhost:5174` in your browser
2. Select a video from local storage or upload one
3. Navigate to any frame using the slider
4. Draw a polygon region of interest (ROI) by clicking points on the frame
5. Choose analysis mode:
   - **Quick Mode**: Single frame detection (~6s)
   - **Full Mode**: Multi-frame tracking with trajectories and CSV export (~60s)
6. Toggle SAHI for improved accuracy (takes longer)
7. View results and download CSV data

## 🎯 Features

- **VisDrone Model**: Specialized for aerial/traffic vehicle detection
- **SAHI Integration**: Sliced detection for small objects (4.7× accuracy boost)
- **Dual Analysis Modes**: 
  - Quick: Instant single-frame detection
  - Full: Track vehicles across time with velocity & trajectory
- **Polygon ROI**: Focus analysis on specific road areas
- **CSV Export**: Vehicle analytics with position, velocity, time data
- **Local Video Support**: Reference large videos without uploading

## 📁 Project Structure

```
├── api_server.py              # Flask REST API
├── interactive_analytics.py   # Core detection & tracking engine
├── frontend/                  # React web interface
│   ├── src/
│   │   └── pages/
│   │       └── AnalysisPage.jsx
│   └── package.json
├── requirements.txt           # Python dependencies
└── README.md
```

## 🛠️ Utilities

- `test_confidence_levels.py` - Test different confidence thresholds
- `test_sahi.py` - Demo SAHI integration
- `estimate_processing_time.py` - Calculate processing time
- `compare_models.py` - Compare model performance

## 📊 Output

Full mode generates `output/vehicle_analytics.csv` with:
- Vehicle ID, class, frames tracked
- Position (start/end x,y in pixels)
- Velocity (pixels per second)
- Total distance traveled
- Time in scene
- Full trajectory points

## ⚙️ Configuration

### Model Settings (in code)
- Confidence threshold: 0.20 (default)
- SAHI slice size: 640×640
- Overlap ratio: 0.2

### Video Requirements
- Format: MP4, AVI, MOV
- Resolution: Any (tested with 1920×1080)
- FPS: Any (tested with 25fps)

## 🐛 Troubleshooting

**Backend won't start:**
- Check if port 5000 is available
- Ensure conda environment is activated
- Install all requirements: `pip install -r requirements.txt`

**Frontend won't start:**
- Delete `node_modules` and run `npm install` again
- Check if port 5174 is available
- Clear `.vite` cache: `rm -rf .vite`

**Model download fails:**
- Model auto-downloads from HuggingFace on first use
- Ensure internet connection
- Clear cache: `rm -rf ~/.cache/huggingface`

**Low detection accuracy:**
- Enable SAHI (increases processing time but improves accuracy)
- Adjust confidence threshold in `api_server.py`
- Ensure video quality is good

## 📝 License

MIT License

## 🤝 Contributing

This is an academic research project. Feel free to fork and modify for your needs.

---

**Note**: The VisDrone model is optimized for vehicle detection in aerial/traffic footage. For best results, use videos with clear vehicle visibility.
