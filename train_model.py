"""
YOLOv8 Training Script for Custom Drone Dataset
Fine-tunes YOLOv8-Nano on custom drone footage for traffic detection
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
from typing import Optional


def create_dataset_yaml(dataset_dir: str, 
                       output_path: str,
                       class_names: list = None) -> str:
    """
    Create dataset.yaml file for YOLO training.
    
    Args:
        dataset_dir: Base directory containing train/val/test splits
        output_path: Path to save dataset.yaml
        class_names: List of class names
        
    Returns:
        Path to created dataset.yaml
    """
    if class_names is None:
        class_names = ['Car', 'Bike', 'Pedestrian']
    
    dataset_path = Path(dataset_dir).absolute()
    
    dataset_config = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Dataset configuration saved: {output_file}")
    print(f"  Classes: {class_names}")
    print(f"  Number of classes: {len(class_names)}")
    
    return str(output_file)


def prepare_dataset_structure(dataset_dir: str):
    """
    Prepare YOLO dataset directory structure.
    
    Args:
        dataset_dir: Base directory for dataset
    """
    dataset_path = Path(dataset_dir)
    
    # Create directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Dataset structure created at: {dataset_dir}")
    print("  Structure:")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  └── labels/")
    print("      ├── train/")
    print("      ├── val/")
    print("      └── test/")


def split_dataset(source_images: str,
                 source_labels: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1):
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_images: Directory with all images
        source_labels: Directory with all labels
        output_dir: Output directory for splits
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    """
    import random
    
    images_path = Path(source_images)
    labels_path = Path(source_labels)
    output_path = Path(output_dir)
    
    # Get all image files
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    
    print(f"\nSplitting {len(image_files)} images...")
    
    # Shuffle
    random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Copy files
    for split_name, files in splits.items():
        print(f"  {split_name}: {len(files)} images")
        
        for img_file in files:
            # Copy image
            dst_img = output_path / 'images' / split_name / img_file.name
            shutil.copy(img_file, dst_img)
            
            # Copy label
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_path / 'labels' / split_name / label_file.name
                shutil.copy(label_file, dst_label)
    
    print("✓ Dataset split complete")


def train_yolo(dataset_yaml: str,
               model_name: str = 'yolov8n.pt',
               epochs: int = 50,
               imgsz: int = 640,
               batch: int = 16,
               project: str = 'runs/train',
               name: str = 'drone_traffic',
               device: str = '0',
               **kwargs) -> YOLO:
    """
    Train YOLOv8 model on custom dataset.
    
    Args:
        dataset_yaml: Path to dataset.yaml
        model_name: Base model to start from
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory
        name: Experiment name
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        **kwargs: Additional training arguments
        
    Returns:
        Trained YOLO model
    """
    print(f"\n{'='*60}")
    print(f"Starting YOLOv8 Training")
    print(f"{'='*60}")
    print(f"Base model: {model_name}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        patience=10,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every N epochs
        cache=True,  # Cache images for faster training
        plots=True,  # Generate training plots
        verbose=True,
        **kwargs
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Training results: {project}/{name}")
    
    return model


def validate_model(model_path: str, 
                   dataset_yaml: str,
                   device: str = '0') -> dict:
    """
    Validate trained model.
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset.yaml
        device: Device to use
        
    Returns:
        Validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Validating Model")
    print(f"{'='*60}")
    
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, device=device)
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def test_inference(model_path: str,
                  image_path: str,
                  output_path: str,
                  conf_threshold: float = 0.25):
    """
    Test model inference on an image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        output_path: Path to save result
        conf_threshold: Confidence threshold
    """
    print(f"\nTesting inference on: {image_path}")
    
    model = YOLO(model_path)
    results = model.predict(
        image_path,
        conf=conf_threshold,
        save=True,
        project=str(Path(output_path).parent),
        name=Path(output_path).stem
    )
    
    print(f"✓ Inference result saved")
    
    # Print detections
    for result in results:
        boxes = result.boxes
        print(f"  Detected {len(boxes)} objects:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    Class {cls}: {conf:.2f}")


def export_model(model_path: str,
                format: str = 'onnx',
                imgsz: int = 640):
    """
    Export model to different format.
    
    Args:
        model_path: Path to trained model
        format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        imgsz: Input image size
    """
    print(f"\nExporting model to {format}...")
    
    model = YOLO(model_path)
    model.export(format=format, imgsz=imgsz)
    
    print(f"✓ Model exported successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on custom drone dataset"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset structure')
    prepare_parser.add_argument('--dataset-dir', type=str, required=True,
                               help='Base directory for dataset')
    prepare_parser.add_argument('--source-images', type=str,
                               help='Directory with labeled images')
    prepare_parser.add_argument('--source-labels', type=str,
                               help='Directory with YOLO format labels')
    prepare_parser.add_argument('--split', action='store_true',
                               help='Split dataset into train/val/test')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--dataset', type=str, required=True,
                             help='Path to dataset.yaml')
    train_parser.add_argument('--model', type=str, default='yolov8n.pt',
                             help='Base model (default: yolov8n.pt)')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640,
                             help='Image size')
    train_parser.add_argument('--device', type=str, default='0',
                             help='Device (0 for GPU, cpu for CPU)')
    train_parser.add_argument('--project', type=str, default='runs/train',
                             help='Project directory')
    train_parser.add_argument('--name', type=str, default='drone_traffic',
                             help='Experiment name')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model')
    val_parser.add_argument('--dataset', type=str, required=True,
                           help='Path to dataset.yaml')
    val_parser.add_argument('--device', type=str, default='0',
                           help='Device')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test inference')
    test_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    test_parser.add_argument('--image', type=str, required=True,
                            help='Path to test image')
    test_parser.add_argument('--output', type=str, default='output/test',
                            help='Output path')
    test_parser.add_argument('--conf', type=float, default=0.25,
                            help='Confidence threshold')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    export_parser.add_argument('--format', type=str, default='onnx',
                              help='Export format')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_dataset_structure(args.dataset_dir)
        
        if args.split and args.source_images and args.source_labels:
            split_dataset(
                args.source_images,
                args.source_labels,
                args.dataset_dir
            )
        
        # Create dataset.yaml
        create_dataset_yaml(
            args.dataset_dir,
            f"{args.dataset_dir}/dataset.yaml"
        )
        
    elif args.command == 'train':
        train_yolo(
            dataset_yaml=args.dataset,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device
        )
        
    elif args.command == 'validate':
        validate_model(args.model, args.dataset, args.device)
        
    elif args.command == 'test':
        test_inference(args.model, args.image, args.output, args.conf)
        
    elif args.command == 'export':
        export_model(args.model, args.format)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
