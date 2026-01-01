"""
Retrain VisDrone Model
Fine-tune the VisDrone YOLOv8 model on corrected annotations
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
import yaml
from datetime import datetime

class VisdroneTrainer:
    """Fine-tune VisDrone model on corrected annotations"""
    
    def __init__(self, data_yaml, base_model='visdrone'):
        """
        Initialize trainer
        
        Args:
            data_yaml: Path to dataset.yaml
            base_model: Base model to fine-tune ('visdrone' or 'yolov8s/m/l')
        """
        self.data_yaml = Path(data_yaml)
        self.base_model = base_model
        self.model = None
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    def load_base_model(self):
        """Load the base VisDrone model for fine-tuning"""
        print("📦 Loading base model for fine-tuning...")
        
        if self.base_model == 'visdrone':
            # Load pre-trained VisDrone model
            try:
                model_path = hf_hub_download(
                    repo_id="mshamrai/yolov8s-visdrone",
                    filename="best.pt"
                )
                self.model = YOLO(model_path)
                print("✅ Loaded VisDrone pre-trained model")
            except Exception as e:
                print(f"⚠️  Could not load VisDrone model: {e}")
                print("Falling back to YOLOv8s...")
                self.model = YOLO('yolov8s.pt')
        else:
            # Load standard YOLOv8 model
            self.model = YOLO(f'{self.base_model}.pt')
            print(f"✅ Loaded {self.base_model} model")
        
        return self.model
    
    def train(self, epochs=50, batch_size=16, img_size=640, patience=10, 
              save_dir='runs/train', device=None):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            patience: Early stopping patience
            save_dir: Directory to save training results
            device: Device to train on (None for auto-detect)
        
        Returns:
            Training results
        """
        if self.model is None:
            self.load_base_model()
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("\n" + "=" * 60)
        print("🚀 Starting VisDrone Model Fine-Tuning")
        print("=" * 60)
        print(f"Dataset: {self.data_yaml}")
        print(f"Base model: {self.base_model}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Device: {device}")
        print(f"Save directory: {save_dir}")
        print("=" * 60 + "\n")
        
        # Train the model
        results = self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            save=True,
            device=device,
            project=save_dir,
            name=f'visdrone_finetuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,  # Lower learning rate for fine-tuning
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        
        return results
    
    def validate(self, model_path=None):
        """
        Validate the trained model
        
        Args:
            model_path: Path to trained model weights (None to use current model)
        
        Returns:
            Validation results
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        print("\n📊 Validating model...")
        results = model.val(data=str(self.data_yaml))
        
        print("\n" + "=" * 60)
        print("📊 Validation Results")
        print("=" * 60)
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        print("=" * 60)
        
        return results
    
    def export_model(self, model_path, output_path='improved_visdrone.pt'):
        """
        Export the trained model
        
        Args:
            model_path: Path to trained model weights
            output_path: Output path for exported model
        """
        import shutil
        
        output_path = Path(output_path)
        shutil.copy2(model_path, output_path)
        
        print(f"\n✅ Model exported to: {output_path.absolute()}")
        print(f"\n💡 To use this model in your application:")
        print(f"   model = YOLO('{output_path}')")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VisDrone model on corrected annotations')
    parser.add_argument('--data', required=True,
                       help='Path to dataset.yaml')
    parser.add_argument('--base-model', default='visdrone',
                       choices=['visdrone', 'yolov8s', 'yolov8m', 'yolov8l'],
                       help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', default=None,
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--save-dir', default='runs/train',
                       help='Directory to save results')
    parser.add_argument('--export', default='improved_visdrone.pt',
                       help='Export path for final model')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VisdroneTrainer(data_yaml=args.data, base_model=args.base_model)
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        patience=args.patience,
        save_dir=args.save_dir,
        device=args.device
    )
    
    # Validate model
    best_model_path = Path(args.save_dir) / results.save_dir.name / 'weights' / 'best.pt'
    trainer.validate(model_path=best_model_path)
    
    # Export model
    trainer.export_model(best_model_path, args.export)
    
    print("\n" + "=" * 60)
    print("🎉 Fine-Tuning Complete!")
    print("=" * 60)
    print(f"\n📁 Training results: {results.save_dir}")
    print(f"📁 Best model: {best_model_path}")
    print(f"📁 Exported model: {args.export}")
    print("\n💡 Next steps:")
    print("   1. Review training metrics in the results directory")
    print("   2. Test the improved model on new videos")
    print("   3. Compare performance with the original model")
    print("   4. Iterate: collect more corrections and retrain!")

if __name__ == "__main__":
    main()
