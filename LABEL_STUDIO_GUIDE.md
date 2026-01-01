# Label Studio Guide for VisDrone Model Improvement

Complete workflow for improving your VisDrone vehicle classification model through annotation correction and iterative retraining.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Workflow Steps](#workflow-steps)
4. [Detailed Instructions](#detailed-instructions)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This workflow enables you to:

1. **Extract frames** from your traffic videos
2. **Run inference** with your current VisDrone model
3. **Import predictions** to Label Studio as pre-annotations
4. **Correct misclassifications** manually in a user-friendly interface
5. **Export corrected annotations** in YOLO format
6. **Retrain the model** with improved data
7. **Iterate** to continuously improve accuracy

### Why Label Studio?

- ✅ **100% Free** - Open source with no limitations
- ✅ **Pre-annotations** - Model predictions appear automatically
- ✅ **Easy corrections** - Click to fix misclassifications
- ✅ **Batch processing** - Handle hundreds of images efficiently
- ✅ **Export flexibility** - Multiple format support

---

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `label-studio` - Annotation platform
- `label-studio-converter` - Format conversion tools
- `ultralytics` - YOLOv8 framework
- `sahi` - Sliced inference for small objects
- Other required packages

### Step 2: Run Setup Script

```bash
python label_studio_setup.py
```

This will:
- Verify Label Studio installation
- Create project configuration
- Generate startup script (`start_labelstudio.bat`)

---

## Workflow Steps

### 🎯 Quick Reference

```
Video → Extract Frames → Run Inference → Import to Label Studio
         ↓                    ↓                    ↓
    batch_inference.py   predictions.json   import_to_labelstudio.py
                                                    ↓
                                            Correct Annotations
                                                    ↓
                                            Export from Label Studio
                                                    ↓
                                        export_corrected_annotations.py
                                                    ↓
                                            Retrain Model
                                                    ↓
                                            retrain_visdrone.py
                                                    ↓
                                            Improved Model! 🎉
```

---

## Detailed Instructions

### 1️⃣ Extract Frames and Run Inference

Extract frames from your traffic video and run VisDrone model predictions:

```bash
python batch_inference.py --input <video_path> --output inference_output --interval 1.0
```

**Arguments:**
- `--input`: Path to video file or image folder
- `--output`: Output directory for frames and predictions
- `--interval`: Frame extraction interval in seconds (default: 1.0)
- `--max-frames`: Maximum frames to extract (optional)
- `--conf`: Confidence threshold (default: 0.20)
- `--sahi`: Enable SAHI for better small object detection (slower but more accurate)

**Example:**

```bash
# Extract 1 frame per second from video
python batch_inference.py --input D1F1_stab.mp4 --output inference_output --interval 1.0

# Use SAHI for better accuracy
python batch_inference.py --input D1F1_stab.mp4 --output inference_output --interval 1.0 --sahi

# Process existing images
python batch_inference.py --input dataset/images/val --output inference_output
```

**Output:**
- `inference_output/frames/` - Extracted frames
- `inference_output/predictions.json` - Model predictions

---

### 2️⃣ Start Label Studio

Run the startup script:

```bash
start_labelstudio.bat
```

Or manually:

```bash
label-studio start label_studio_project --port 8080
```

Label Studio will open at: **http://localhost:8080**

---

### 3️⃣ Create Label Studio Project

1. **Open browser** to http://localhost:8080
2. **Sign up** (local account, no internet required)
3. **Create new project**:
   - Name: "VisDrone Vehicle Correction"
   - Click "Data Import" → Skip for now
4. **Configure labeling interface**:
   - Go to "Settings" → "Labeling Interface"
   - Click "Code" and paste the configuration from `label_studio_project/project_config.json`
   - Or use the visual editor to add:
     - Image display
     - Rectangle labels with VisDrone classes

---

### 4️⃣ Set Up Local File Storage

To access your extracted frames:

1. Go to **Settings** → **Cloud Storage**
2. Click **Add Source Storage**
3. Select **Local Files**
4. Set path to: `C:\Users\sakth\Documents\Projects\traffique_v2\inference_output\frames`
5. Click **Add Storage**
6. Click **Sync Storage**

---

### 5️⃣ Import Predictions to Label Studio

Convert predictions to Label Studio format:

```bash
python import_to_labelstudio.py --predictions inference_output --output labelstudio_import.json
```

**Arguments:**
- `--predictions`: Path to predictions directory
- `--output`: Output import file path
- `--image-path`: Custom base path for images (optional)
- `--project-id`: Project ID for direct API import (optional)

**Import to Label Studio:**

1. In Label Studio, go to your project
2. Click **Import**
3. Upload `labelstudio_import.json`
4. Predictions will appear as **pre-annotations**!

---

### 6️⃣ Correct Annotations

Now the fun part! Review and correct the model's predictions:

#### Navigation
- Use **arrow keys** or click **Next/Previous** to navigate images
- Press **Space** to submit and move to next task

#### Correcting Classifications
1. **Click on a bounding box** to select it
2. **Change the label** using the dropdown or keyboard shortcuts
3. **Adjust box** by dragging corners if needed
4. **Delete** false positives by selecting and pressing Delete

#### Adding Missing Vehicles
1. Click **Rectangle** tool
2. Select the **vehicle class**
3. **Draw bounding box** around the vehicle

#### Keyboard Shortcuts
- `1-9`: Quick select class labels
- `Delete`: Remove selected box
- `Ctrl+Z`: Undo
- `Ctrl+Enter`: Submit task
- `Space`: Submit and next

#### Tips
- Focus on **obvious misclassifications** first
- Don't worry about perfect boxes - close enough is fine
- **Skip unclear/occluded vehicles** if uncertain
- Aim for **quality over quantity**

---

### 7️⃣ Export Corrected Annotations

Once you've corrected annotations:

1. In Label Studio, click **Export**
2. Select **JSON** format
3. Download the export file (e.g., `project-1-export.json`)

---

### 8️⃣ Convert to YOLO Format

Convert Label Studio export to YOLO training format:

```bash
python export_corrected_annotations.py --export project-1-export.json --output training_dataset
```

**Arguments:**
- `--export`: Path to Label Studio JSON export
- `--output`: Output directory for YOLO dataset
- `--train-split`: Training/validation split ratio (default: 0.8)

**Output:**
```
training_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

---

### 9️⃣ Retrain VisDrone Model

Fine-tune the VisDrone model on your corrected annotations:

```bash
python retrain_visdrone.py --data training_dataset/dataset.yaml --epochs 50 --batch 16
```

**Arguments:**
- `--data`: Path to dataset.yaml
- `--base-model`: Base model (visdrone, yolov8s, yolov8m, yolov8l)
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--patience`: Early stopping patience (default: 10)
- `--device`: Training device (cuda/cpu)
- `--export`: Export path for final model (default: improved_visdrone.pt)

**Training Tips:**
- Start with **50 epochs** for initial training
- Use **batch size 16** if you have 8GB+ GPU
- Reduce batch size to **8 or 4** if you get out-of-memory errors
- Training will **auto-stop** if no improvement for 10 epochs

**Output:**
- `runs/train/visdrone_finetuned_*/` - Training results
- `runs/train/visdrone_finetuned_*/weights/best.pt` - Best model
- `improved_visdrone.pt` - Exported model ready to use

---

### 🔟 Use Improved Model

Update your application to use the improved model:

```python
from ultralytics import YOLO

# Load your improved model
model = YOLO('improved_visdrone.pt')

# Use it for inference
results = model.predict('video.mp4', conf=0.20)
```

Or update `interactive_analytics.py` to use the new model by default.

---

## Best Practices

### 🎯 Active Learning Strategy

1. **Start small**: Correct 50-100 images first
2. **Retrain**: See improvement quickly
3. **Focus on errors**: Prioritize correcting common mistakes
4. **Iterate**: Repeat the cycle to continuously improve

### 📊 Data Quality

- **Diverse scenarios**: Include different times, weather, angles
- **Edge cases**: Focus on difficult examples (occlusion, small vehicles)
- **Balanced classes**: Ensure all vehicle types are represented
- **Consistent labeling**: Use the same criteria for all annotations

### ⚡ Efficiency Tips

- **Batch correction**: Do 20-30 images in one session
- **Use keyboard shortcuts**: Much faster than mouse
- **Skip perfect predictions**: Only correct mistakes
- **Set confidence threshold**: Higher threshold = fewer false positives to delete

### 🔄 Iteration Cycle

```
Annotate 50 images → Train → Test → Identify weak areas
                                          ↓
                    ← Annotate more examples ←
```

Each iteration should improve specific weaknesses.

---

## Troubleshooting

### Label Studio won't start

**Error**: Port 8080 already in use

**Solution**:
```bash
label-studio start label_studio_project --port 8081
```

### Images not showing in Label Studio

**Problem**: Local storage path incorrect

**Solution**:
1. Check the path in Settings → Cloud Storage
2. Use **absolute path**: `C:\Users\sakth\Documents\Projects\traffique_v2\inference_output\frames`
3. Click **Sync Storage** after updating

### Predictions not appearing

**Problem**: Import file format incorrect

**Solution**:
1. Verify `predictions.json` exists in inference output
2. Re-run `import_to_labelstudio.py`
3. Check image paths in the import JSON file

### Training fails with CUDA out of memory

**Solution**:
```bash
# Reduce batch size
python retrain_visdrone.py --data training_dataset/dataset.yaml --batch 8

# Or use CPU (slower)
python retrain_visdrone.py --data training_dataset/dataset.yaml --device cpu
```

### Model not improving

**Possible causes**:
1. **Too few annotations**: Need at least 50-100 corrected images
2. **Unbalanced classes**: Ensure all vehicle types are represented
3. **Low quality corrections**: Review annotation consistency
4. **Learning rate too high**: Model already uses optimized settings

**Solution**: Annotate more diverse examples and retrain

### Export fails

**Problem**: Label Studio export is empty

**Solution**:
1. Ensure you've **submitted** annotations (not just saved)
2. Check that tasks show as "Completed" in Label Studio
3. Try exporting as JSON-MIN format instead

---

## Summary

You now have a complete workflow to:

✅ Extract frames from videos  
✅ Run model inference  
✅ Import to Label Studio with pre-annotations  
✅ Correct misclassifications easily  
✅ Export in YOLO format  
✅ Retrain and improve your model  
✅ Iterate continuously  

### Quick Command Reference

```bash
# 1. Setup
python label_studio_setup.py

# 2. Extract and infer
python batch_inference.py --input video.mp4 --output inference_output

# 3. Start Label Studio
start_labelstudio.bat

# 4. Import predictions
python import_to_labelstudio.py --predictions inference_output

# 5. (Correct in Label Studio UI, then export)

# 6. Convert to YOLO
python export_corrected_annotations.py --export project-1-export.json

# 7. Retrain
python retrain_visdrone.py --data training_dataset/dataset.yaml

# 8. Use improved model!
```

---

## Need Help?

- **Label Studio Docs**: https://labelstud.io/guide/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **VisDrone Dataset**: http://aiskyeye.com/

Happy annotating! 🚗🚙🚌
