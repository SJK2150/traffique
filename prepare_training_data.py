#!/usr/bin/env python3
"""
Prepare Label Studio exports for YOLOv8 training

After exporting from Label Studio in YOLO format:
1. Extract the zip file
2. Run this script to organize data for training
"""

import shutil
from pathlib import Path
import yaml

def prepare_training_data(export_dir, output_dir="training_data"):
    """
    Organize Label Studio YOLO export for training
    
    Args:
        export_dir: Path to extracted Label Studio export
        output_dir: Output directory for organized training data
    """
    export_path = Path(export_dir)
    output_path = Path(output_dir)
    
    print("📁 Preparing training data...")
    
    # Create directory structure
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Find images and labels in export
    images = list(export_path.glob("**/*.jpg")) + list(export_path.glob("**/*.png"))
    labels = list(export_path.glob("**/*.txt"))
    
    print(f"   Found {len(images)} images")
    print(f"   Found {len(labels)} label files")
    
    # Split 80/20 train/val
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy training images and labels
    print("\n📋 Copying training data...")
    for img_path in train_images:
        # Copy image
        shutil.copy(img_path, output_path / "images" / "train" / img_path.name)
        
        # Copy corresponding label
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, output_path / "labels" / "train" / label_path.name)
    
    # Copy validation images and labels
    print("📋 Copying validation data...")
    for img_path in val_images:
        shutil.copy(img_path, output_path / "images" / "val" / img_path.name)
        
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, output_path / "labels" / "val" / label_path.name)
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'Pedestrian',
            1: 'People',
            2: 'Bicycle',
            3: 'Car',
            4: 'Van',
            5: 'Truck',
            6: 'Tricycle',
            7: 'Awning-tricycle',
            8: 'Bus',
            9: 'Motor'
        }
    }
    
    with open(output_path / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\n✅ Training data prepared!")
    print(f"   Train images: {len(train_images)}")
    print(f"   Val images: {len(val_images)}")
    print(f"   Dataset config: {output_path / 'dataset.yaml'}")
    
    return output_path / "dataset.yaml"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Label Studio export for training")
    parser.add_argument("--export-dir", required=True, help="Path to extracted Label Studio export")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    
    args = parser.parse_args()
    
    prepare_training_data(args.export_dir, args.output_dir)
