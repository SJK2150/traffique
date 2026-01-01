# VisDrone Fine-tuning Guide

## Overview
This guide explains how to fine-tune the VisDrone YOLOv8 model with your corrected Label Studio annotations to improve detection of small vehicles (bikes, motors, etc.).

## Dataset Summary
- **Total annotated frames**: 10
- **Training set**: 8 images (with 102+ annotations per image)
- **Validation set**: 2 images
- **Classes**: 10 VisDrone classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)

## Files Created

### 1. `convert_labelstudio_to_yolo.py`
Converts Label Studio export to YOLO format:
- Extracts images from base64 data
- Converts bounding box annotations to YOLO format
- Creates train/val split (80/20)
- Generates `dataset.yaml`

### 2. `finetune_model.py`
Main fine-tuning script with optimized hyperparameters:
- **Base model**: `mshamrai/yolov8s-visdrone`
- **Epochs**: 50 (with early stopping)
- **Batch size**: 8
- **Learning rate**: 0.001 (lower for fine-tuning)
- **Augmentation**: Light augmentation suitable for fine-tuning
- **Optimizer**: AdamW

### 3. `verify_dataset.py`
Verifies the dataset structure before training.

## How to Fine-tune

### Step 1: Verify Dataset (Already Done ✅)
```bash
python verify_dataset.py
```

### Step 2: Start Fine-tuning
```bash
python finetune_model.py
```

**Note**: This will take some time depending on your hardware:
- **With GPU**: ~10-20 minutes
- **Without GPU (CPU only)**: 1-2 hours

### Step 3: Monitor Training
The script will show:
- Training progress with loss metrics
- Validation metrics (mAP50, mAP50-95)
- Best model checkpoint

Results are saved to: `runs/finetune/visdrone_finetuned/`

## Training Parameters Explained

### Why These Settings?
1. **Lower Learning Rate (0.001)**: Since we're fine-tuning, not training from scratch
2. **Light Augmentation**: Preserves the specific characteristics of your traffic footage
3. **Early Stopping (patience=10)**: Prevents overfitting on small dataset
4. **AdamW Optimizer**: Better for fine-tuning with weight decay

### Adjusting Parameters
If you have more data later or want to experiment:

```python
# In finetune_model.py, adjust these:
epochs=100,        # More epochs for larger datasets
batch_size=16,     # Larger batch if you have more GPU memory
lr0=0.0001,        # Even lower LR for very careful fine-tuning
```

## After Training

### Using the Fine-tuned Model
```python
from ultralytics import YOLO

# Load your fine-tuned model
model = YOLO('visdrone_finetuned.pt')

# Run inference
results = model.predict('your_video.mp4', conf=0.25)
```

### Compare with Base Model
```python
from ultralytics import YOLO

# Base model
base_model = YOLO('mshamrai/yolov8s-visdrone')
base_results = base_model.predict('test_video.mp4', conf=0.25)

# Fine-tuned model
finetuned_model = YOLO('visdrone_finetuned.pt')
finetuned_results = finetuned_model.predict('test_video.mp4', conf=0.25)

# Compare detections, especially for bikes and motors
```

## Expected Improvements
After fine-tuning on your corrected annotations, you should see:
- ✅ Better detection of small vehicles (bikes, motors)
- ✅ Fewer false positives
- ✅ More accurate bounding boxes
- ✅ Better performance on your specific traffic footage

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```python
batch_size=4,  # or even 2
```

### Training Too Slow
- Use GPU if available
- Reduce image size: `imgsz=416` instead of 640
- Reduce epochs: `epochs=30`

### Overfitting (val loss increases)
- Increase augmentation
- Add more annotated data
- Use early stopping (already enabled)

## Next Steps

1. **Annotate More Data**: 10 images is a good start, but 50-100 would be better
2. **Test on Full Video**: Run the fine-tuned model on your full traffic footage
3. **Iterative Improvement**: 
   - Find frames where model still fails
   - Annotate those frames
   - Re-train with expanded dataset

## File Structure
```
traffique_v2/
├── visdrone_finetuning/          # YOLO dataset
│   ├── images/
│   │   ├── train/                # 8 training images
│   │   └── val/                  # 2 validation images
│   ├── labels/
│   │   ├── train/                # 8 training labels
│   │   └── val/                  # 2 validation labels
│   └── dataset.yaml              # Dataset configuration
├── runs/finetune/visdrone_finetuned/  # Training results
│   ├── weights/
│   │   ├── best.pt               # Best model checkpoint
│   │   └── last.pt               # Last epoch checkpoint
│   ├── results.png               # Training curves
│   └── confusion_matrix.png      # Confusion matrix
└── visdrone_finetuned.pt         # Final model (copy of best.pt)
```

## Ready to Train!
Everything is set up. Just run:
```bash
python finetune_model.py
```
