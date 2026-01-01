"""
Export Corrected Annotations from Label Studio
Converts Label Studio exports to YOLO training format
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

class LabelStudioExporter:
    """Export and convert Label Studio annotations to YOLO format"""
    
    def __init__(self):
        """Initialize exporter"""
        # VisDrone class mapping
        self.class_names = {
            'pedestrian': 0,
            'people': 1,
            'bicycle': 2,
            'car': 3,
            'van': 4,
            'truck': 5,
            'tricycle': 6,
            'awning-tricycle': 7,
            'bus': 8,
            'motor': 9
        }
    
    def convert_labelstudio_to_yolo(self, labelstudio_export, output_dir, train_split=0.8):
        """
        Convert Label Studio export to YOLO format
        
        Args:
            labelstudio_export: Path to Label Studio JSON export file
            output_dir: Output directory for YOLO dataset
            train_split: Fraction of data for training (rest for validation)
        
        Returns:
            Dictionary with dataset statistics
        """
        output_dir = Path(output_dir)
        
        # Create YOLO directory structure
        train_images_dir = output_dir / 'images' / 'train'
        train_labels_dir = output_dir / 'labels' / 'train'
        val_images_dir = output_dir / 'images' / 'val'
        val_labels_dir = output_dir / 'labels' / 'val'
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load Label Studio export
        with open(labelstudio_export, 'r') as f:
            annotations = json.load(f)
        
        print(f"📝 Converting {len(annotations)} annotations to YOLO format...")
        
        # Split into train/val
        num_train = int(len(annotations) * train_split)
        train_annotations = annotations[:num_train]
        val_annotations = annotations[num_train:]
        
        stats = {
            'total': len(annotations),
            'train': len(train_annotations),
            'val': len(val_annotations),
            'class_counts': {name: 0 for name in self.class_names.keys()}
        }
        
        # Process training set
        print("\n📦 Processing training set...")
        self._process_annotations(train_annotations, train_images_dir, train_labels_dir, stats)
        
        # Process validation set
        print("\n📦 Processing validation set...")
        self._process_annotations(val_annotations, val_images_dir, val_labels_dir, stats)
        
        # Create dataset.yaml
        self._create_dataset_yaml(output_dir)
        
        return stats
    
    def _process_annotations(self, annotations, images_dir, labels_dir, stats):
        """Process a set of annotations"""
        
        for task in tqdm(annotations, desc="Converting"):
            # Get image info
            image_path = self._extract_image_path(task)
            if not image_path or not Path(image_path).exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            # Get annotations (use corrected annotations if available, else predictions)
            if 'annotations' in task and len(task['annotations']) > 0:
                # Use human-corrected annotations
                results = task['annotations'][0]['result']
            elif 'predictions' in task and len(task['predictions']) > 0:
                # Use model predictions (if not corrected)
                results = task['predictions'][0]['result']
            else:
                # No annotations
                continue
            
            # Get image dimensions
            if 'annotations' in task and len(task['annotations']) > 0:
                img_width = task['annotations'][0].get('original_width', 1920)
                img_height = task['annotations'][0].get('original_height', 1080)
            else:
                # Try to get from image file
                import cv2
                img = cv2.imread(str(image_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                else:
                    img_width, img_height = 1920, 1080
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            
            for result in results:
                if result['type'] != 'rectanglelabels':
                    continue
                
                # Get class name
                class_name = result['value']['rectanglelabels'][0]
                if class_name not in self.class_names:
                    continue
                
                class_id = self.class_names[class_name]
                stats['class_counts'][class_name] += 1
                
                # Get bounding box (in percentage)
                x_percent = result['value']['x']
                y_percent = result['value']['y']
                width_percent = result['value']['width']
                height_percent = result['value']['height']
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x_percent + width_percent / 2) / 100
                y_center = (y_percent + height_percent / 2) / 100
                width_norm = width_percent / 100
                height_norm = height_percent / 100
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Skip if no valid annotations
            if not yolo_annotations:
                continue
            
            # Copy image
            image_filename = Path(image_path).name
            dest_image = images_dir / image_filename
            shutil.copy2(image_path, dest_image)
            
            # Save YOLO label file
            label_filename = Path(image_path).stem + '.txt'
            label_file = labels_dir / label_filename
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    def _extract_image_path(self, task):
        """Extract image path from Label Studio task"""
        if 'data' in task and 'image' in task['data']:
            image_ref = task['data']['image']
            
            # Handle different image reference formats
            if image_ref.startswith('/data/local-files/?d='):
                # Local file reference
                return image_ref.replace('/data/local-files/?d=', '')
            elif image_ref.startswith('http'):
                # URL reference (not supported for local training)
                return None
            else:
                return image_ref
        
        return None
    
    def _create_dataset_yaml(self, output_dir):
        """Create dataset.yaml for YOLO training"""
        
        yaml_content = f"""# VisDrone Vehicle Detection Dataset
# Generated from Label Studio corrected annotations

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
        
        yaml_file = output_dir / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Dataset configuration saved: {yaml_file}")

def main():
    parser = argparse.ArgumentParser(description='Export Label Studio annotations to YOLO format')
    parser.add_argument('--export', required=True,
                       help='Path to Label Studio JSON export file')
    parser.add_argument('--output', default='training_dataset',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = LabelStudioExporter()
    
    # Convert annotations
    stats = exporter.convert_labelstudio_to_yolo(
        args.export,
        args.output,
        train_split=args.train_split
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Conversion Summary")
    print("=" * 60)
    print(f"Total annotations: {stats['total']}")
    print(f"Training set: {stats['train']}")
    print(f"Validation set: {stats['val']}")
    print(f"\nClass distribution:")
    for class_name, count in stats['class_counts'].items():
        if count > 0:
            print(f"  {class_name}: {count}")
    print("=" * 60)
    print(f"\n✅ Dataset ready for training!")
    print(f"   Location: {Path(args.output).absolute()}")
    print(f"\n📝 Next step: python retrain_visdrone.py --data {args.output}/dataset.yaml")

if __name__ == "__main__":
    main()
