"""
Prepare the fine-tuning dataset from Label Studio export.
This script:
1. Extracts images from the base64 JSON
2. Copies the corrected annotations from Label Studio export
3. Creates a proper YOLO dataset structure with train/val split
4. Generates a dataset.yaml configuration file
"""

import json
import base64
import shutil
from pathlib import Path
from PIL import Image
import io

def prepare_finetuning_dataset(
    json_path="label_studio_embedded.json",
    export_dir="label_studio_export",
    output_dir="visdrone_finetuning",
    val_split=0.2
):
    """
    Prepare the dataset for YOLOv8 fine-tuning.
    
    Args:
        json_path: Path to the base64 embedded JSON file
        export_dir: Path to Label Studio export directory
        output_dir: Output directory for the fine-tuning dataset
        val_split: Fraction of data to use for validation (0.2 = 20%)
    """
    
    # Create output directory structure
    output_path = Path(output_dir)
    train_images = output_path / "images" / "train"
    val_images = output_path / "images" / "val"
    train_labels = output_path / "labels" / "train"
    val_labels = output_path / "labels" / "val"
    
    for dir_path in [train_images, val_images, train_labels, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset structure in {output_path}")
    
    # Load the base64 JSON to extract images
    print(f"\nLoading images from {json_path}...")
    with open(json_path, 'r') as f:
        tasks = json.load(f)
    
    # Extract images and save them
    image_mapping = {}  # Maps base64 filename to actual frame number
    
    for task in tasks:
        # Extract image from base64
        image_data = task['data']['image']
        if image_data.startswith('data:image'):
            # Format: data:image/jpeg;base64,<base64_string>
            base64_str = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Get the frame number from the task (if available in annotations)
            # We'll use the base64 string as a unique identifier
            base64_filename = base64.b64encode(base64_str[:20].encode()).decode()[:10]
            
            # Try to find a better name from the predictions
            if 'predictions' in task and len(task['predictions']) > 0:
                # Check if there's metadata about the original filename
                # For now, we'll just use sequential numbering
                pass
            
            # Save with a clean filename
            frame_num = len(image_mapping)
            clean_filename = f"frame_{frame_num:04d}.jpg"
            image_mapping[base64_filename] = clean_filename
            
    print(f"Found {len(image_mapping)} images")
    
    # Now process the Label Studio export
    export_path = Path(export_dir)
    labels_path = export_path / "labels"
    
    if not labels_path.exists():
        print(f"\nError: Labels directory not found at {labels_path}")
        print("Please make sure you've exported the annotations from Label Studio.")
        return
    
    # Get all label files
    label_files = list(labels_path.glob("*.txt"))
    print(f"\nFound {len(label_files)} annotated files")
    
    if len(label_files) == 0:
        print("No annotations found! Please check the export directory.")
        return
    
    # Re-extract images and match with labels
    print("\nExtracting images and matching with annotations...")
    processed_count = 0
    
    # We need to match the base64 filenames from Label Studio export
    # with our images. Let's reload and process properly.
    
    # First, let's just extract all images with their original base64 names
    temp_images = {}
    for task in tasks:
        image_data = task['data']['image']
        if image_data.startswith('data:image'):
            base64_str = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Use the same base64 encoding that Label Studio uses for filenames
            # The filename is the base64 of the image data URL
            img_data_url = task['data']['image']
            filename_base64 = base64.b64encode(img_data_url.encode()).decode()
            
            temp_images[filename_base64] = img
    
    print(f"Extracted {len(temp_images)} images")
    print(f"Label files: {[f.stem for f in label_files]}")
    
    # Match and copy files
    matched_files = []
    for label_file in label_files:
        base64_name = label_file.stem  # e.g., "2Q=="
        
        # Find matching image
        matching_image = None
        for img_key, img in temp_images.items():
            if base64_name in img_key or img_key.endswith(base64_name):
                matching_image = img
                break
        
        if matching_image is None:
            # Try direct match
            if base64_name in temp_images:
                matching_image = temp_images[base64_name]
        
        if matching_image:
            matched_files.append((matching_image, label_file))
        else:
            print(f"Warning: Could not find image for label {base64_name}")
    
    print(f"\nMatched {len(matched_files)} image-label pairs")
    
    # Split into train and val
    import random
    random.seed(42)
    random.shuffle(matched_files)
    
    val_count = int(len(matched_files) * val_split)
    val_files = matched_files[:val_count]
    train_files = matched_files[val_count:]
    
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    
    # Copy files
    print("\nCopying files...")
    
    for idx, (img, label_file) in enumerate(train_files):
        filename = f"frame_{idx:04d}.jpg"
        img.save(train_images / filename)
        shutil.copy(label_file, train_labels / f"frame_{idx:04d}.txt")
    
    for idx, (img, label_file) in enumerate(val_files):
        filename = f"frame_{idx:04d}.jpg"
        img.save(val_images / filename)
        shutil.copy(label_file, val_labels / f"frame_{idx:04d}.txt")
    
    print("Files copied successfully!")
    
    # Create dataset.yaml
    print("\nCreating dataset.yaml...")
    
    # Read class names from the export
    classes_file = export_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        # Default VisDrone classes
        class_names = [
            'Awning-tricycle', 'Bicycle', 'Bus', 'Car', 'Motor',
            'Pedestrian', 'Tricycle', 'Truck', 'Van'
        ]
    
    # Create absolute paths for the dataset
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"# VisDrone Fine-tuning Dataset\n")
        f.write(f"# Generated from Label Studio annotations\n\n")
        f.write(f"path: {dataset_yaml['path']}\n")
        f.write(f"train: {dataset_yaml['train']}\n")
        f.write(f"val: {dataset_yaml['val']}\n\n")
        f.write(f"nc: {dataset_yaml['nc']}\n")
        f.write(f"names: {class_names}\n")
    
    print(f"Dataset configuration saved to {yaml_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Classes: {len(class_names)}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/  ({len(train_files)} images)")
    print(f"    │   └── val/    ({len(val_files)} images)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/  ({len(train_files)} labels)")
    print(f"    │   └── val/    ({len(val_files)} labels)")
    print(f"    └── dataset.yaml")
    print("\nYou can now use this dataset to fine-tune YOLOv8!")
    print(f"Dataset config: {yaml_path}")

if __name__ == "__main__":
    prepare_finetuning_dataset()
