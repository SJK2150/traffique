"""
Prepare the fine-tuning dataset from Label Studio export.
Simplified version that extracts images from JSON and matches them with exported labels.
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
    print(f"\nLoading tasks from {json_path}...")
    with open(json_path, 'r') as f:
        tasks = json.load(f)
    
    print(f"Found {len(tasks)} tasks")
    
    # Extract all images with their task IDs
    images_by_task_id = {}
    
    for task in tasks:
        task_id = task.get('id')
        image_data = task['data']['image']
        
        if image_data.startswith('data:image'):
            # Format: data:image/jpeg;base64,<base64_string>
            base64_str = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_bytes))
            images_by_task_id[task_id] = img
    
    print(f"Extracted {len(images_by_task_id)} images")
    
    # Now we need to export from Label Studio again, but this time with task IDs
    # For now, let's use a simpler approach: match by order
    
    export_path = Path(export_dir)
    labels_path = export_path / "labels"
    
    if not labels_path.exists():
        print(f"\nError: Labels directory not found at {labels_path}")
        return
    
    # Get all label files
    label_files = sorted(list(labels_path.glob("*.txt")))
    print(f"\nFound {len(label_files)} annotated files")
    
    if len(label_files) == 0:
        print("No annotations found!")
        return
    
    # Since we can't easily match base64 filenames, let's use the actual
    # Label Studio export JSON if it exists
    export_json_path = export_path.parent / "project-1-at-2025-01-23-07-09-b0b8b0c2.json"
    
    if not export_json_path.exists():
        # Try to find any JSON file in the parent directory
        json_files = list(export_path.parent.glob("project-*.json"))
        if json_files:
            export_json_path = json_files[0]
            print(f"\nFound export JSON: {export_json_path.name}")
        else:
            print("\nWarning: Could not find Label Studio export JSON")
            print("Will try to match by extracting images from embedded JSON")
            
            # Fallback: just use all images in order
            all_images = list(images_by_task_id.values())
            
            if len(all_images) < len(label_files):
                print(f"Error: Have {len(label_files)} labels but only {len(all_images)} images")
                return
            
            matched_files = [(all_images[i], label_files[i]) for i in range(len(label_files))]
    else:
        # Load the export JSON to get proper task mapping
        print(f"Loading export JSON: {export_json_path.name}")
        with open(export_json_path, 'r') as f:
            export_data = json.load(f)
        
        print(f"Found {len(export_data)} exported tasks")
        
        # Match images with annotations
        matched_files = []
        
        for export_task in export_data:
            task_id = export_task.get('id')
            
            # Get the image for this task
            if task_id in images_by_task_id:
                img = images_by_task_id[task_id]
                
                # Find the corresponding label file
                # The label filename is the base64 of the image data URL
                image_url = export_task['data']['image']
                filename_base64 = base64.b64encode(image_url.encode()).decode()
                
                # Try to find matching label file
                label_file = None
                for lf in label_files:
                    if lf.stem == filename_base64 or filename_base64.startswith(lf.stem):
                        label_file = lf
                        break
                
                if label_file:
                    matched_files.append((img, label_file))
                else:
                    print(f"Warning: No label file found for task {task_id}")
        
        if len(matched_files) == 0:
            print("\nCould not match files using export JSON. Trying simple matching...")
            # Fallback to simple matching
            all_images = list(images_by_task_id.values())
            matched_files = [(all_images[i], label_files[i]) for i in range(min(len(all_images), len(label_files)))]
    
    print(f"\nMatched {len(matched_files)} image-label pairs")
    
    if len(matched_files) == 0:
        print("Error: No matched files!")
        return
    
    # Split into train and val
    import random
    random.seed(42)
    random.shuffle(matched_files)
    
    val_count = max(1, int(len(matched_files) * val_split))
    val_files = matched_files[:val_count]
    train_files = matched_files[val_count:]
    
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    
    # Copy files
    print("\nCopying files...")
    
    for idx, (img, label_file) in enumerate(train_files):
        filename = f"frame_{idx:04d}.jpg"
        img.save(train_images / filename, quality=95)
        shutil.copy(label_file, train_labels / f"frame_{idx:04d}.txt")
    
    for idx, (img, label_file) in enumerate(val_files):
        filename = f"frame_{idx:04d}.jpg"
        img.save(val_images / filename, quality=95)
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
