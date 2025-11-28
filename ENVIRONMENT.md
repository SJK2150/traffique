# Environment Setup Guide

## ✅ Installation Complete

Your CUDA-enabled Python environment is ready to use!

### Environment Details
- **Name**: `iitmlab`
- **Python**: 3.11
- **PyTorch**: 2.5.1+cu121 (CUDA 12.1)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **Status**: torch.cuda.is_available() = **True** ✓

### Installed Packages
- PyTorch 2.5.1 (with CUDA 12.1 support)
- Ultralytics 8.3.233 (YOLOv8)
- OpenCV 4.12.0.88 (with contrib)
- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- FilterPy, PyYAML, tqdm

---

## How to Use This Environment

### Option 1: Activate in Current Terminal (Recommended)
```powershell
# Activate the environment
conda activate iitmlab

# Verify GPU is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run your scripts
python main.py
python train_model.py
```

### Option 2: Direct Python Execution (No Activation Needed)
```powershell
# Run any Python script directly using the env's Python
C:\Users\sakth\miniconda3\envs\iitmlab\python.exe main.py
C:\Users\sakth\miniconda3\envs\iitmlab\python.exe train_model.py

# Or shorter version
%USERPROFILE%\miniconda3\envs\iitmlab\python.exe main.py
```

### Option 3: Use in VS Code
1. Open Command Palette: `Ctrl+Shift+P`
2. Type: **"Python: Select Interpreter"**
3. Choose: `Python 3.11.14 ('iitmlab')` from `C:\Users\sakth\miniconda3\envs\iitmlab\python.exe`
4. All Python scripts will now use this environment automatically

---

## Quick Verification Commands

### Check GPU Status
```powershell
conda activate iitmlab
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Check Ultralytics
```powershell
conda activate iitmlab
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"
```

### List Installed Packages
```powershell
conda activate iitmlab
pip list
```

---

## Common Commands

### Running the Full System
```powershell
conda activate iitmlab
python main.py
```

### Training a Model
```powershell
conda activate iitmlab
python train_model.py
```

### Extract Frames from Videos
```powershell
conda activate iitmlab
python extract_frames.py
```

### Interactive Calibration Tool
```powershell
conda activate iitmlab
python calibration.py
```

### Analyze Results
```powershell
conda activate iitmlab
python analyze_results.py
```

---

## Troubleshooting

### If conda command not found:
Close and reopen PowerShell, or manually add to PATH:
```powershell
$env:PATH = "$env:USERPROFILE\miniconda3\Scripts;$env:USERPROFILE\miniconda3;$env:PATH"
```

### If GPU not detected after activation:
1. Ensure NVIDIA drivers are up to date (you have 566.07 - good!)
2. Restart your computer
3. Re-run verification:
   ```powershell
   conda activate iitmlab
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Installing Additional Packages
```powershell
conda activate iitmlab
pip install package-name
```

---

## Environment Info
- **Location**: `C:\Users\sakth\miniconda3\envs\iitmlab`
- **Python Executable**: `C:\Users\sakth\miniconda3\envs\iitmlab\python.exe`
- **Created**: November 27, 2025
- **CUDA Compatibility**: 12.1 (compatible with your driver 12.7)

---

## Next Steps
1. ✅ Environment is ready
2. 📁 Prepare your dataset (see `QUICKSTART.md`)
3. 🎯 Extract frames: `python extract_frames.py`
4. 📐 Calibrate cameras: `python calibration.py`
5. 🚀 Train model: `python train_model.py`
6. ▶️ Run system: `python main.py`

**Your GPU will now be used for inference and training!** 🚀
