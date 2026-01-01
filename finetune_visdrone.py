#!/usr/bin/env python3
"""
Fine-tune VisDrone YOLOv8 model with corrected annotations

This will improve detection accuracy, especially for bikes/motors!
"""

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

def finetune_visdrone(dataset_yaml, epochs=50, batch_size=8, img_size=640):
    """
    Fine-tune VisDrone model with corrected annotations
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
    """
    
    print("="*70)
    print("  FINE-TUNING VISDRONE MODEL")
    print("="*70)
    
    # Load pre-trained VisDrone model
    print("\n📥 Loading VisDrone base model...")
    try:
        model_path = hf_hub_download(
            repo_id="mshamrai/yolov8s-visdrone",
            filename="best.pt"
        )
        print("   ✓ Downloaded from HuggingFace")
    except:
        model_path = "best.pt"
        print("   ⚠️  Using local model")
    
    model = YOLO(model_path)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    # Training configuration
    print(f"\n⚙️  Training configuration:")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    
    # Start training
    print(f"\n🚀 Starting fine-tuning...")
    print("   This may take a while depending on your GPU...")
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="visdrone_finetuned",
        name="run",
        patience=10,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n✅ Fine-tuning complete!")
    print(f"   Best model saved to: visdrone_finetuned/run/weights/best.pt")
    print(f"   Training plots saved to: visdrone_finetuned/run/")
    
    # Validate the model
    print("\n📊 Validating model...")
    metrics = model.val()
    
    print(f"\n📈 Validation Results:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    
    return "visdrone_finetuned/run/weights/best.pt"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune VisDrone model")
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    finetune_visdrone(
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
