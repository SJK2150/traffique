"""
Fine-tune YOLOv8 on VisDrone dataset with your corrected annotations.
This script will:
1. Load the pre-trained VisDrone model
2. Fine-tune it on your annotated data
3. Save the improved model
"""

from ultralytics import YOLO
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

def finetune_visdrone(
    base_model="mshamrai/yolov8s-visdrone",
    data_yaml="visdrone_finetuning/dataset.yaml",
    epochs=50,
    batch_size=8,
    imgsz=640,
    output_name="visdrone_finetuned"
):
    """
    Fine-tune the VisDrone model on your corrected annotations.
    
    Args:
        base_model: Base model to fine-tune from
        data_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
        output_name: Name for the output model
    """
    print(f"🚀 Starting fine-tuning process...")
    print(f"   Base model: {base_model}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {imgsz}")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    # Download the base model from Hugging Face if needed
    print(f"\n📥 Downloading base model from Hugging Face...")
    local_model_path = Path("yolov8s-visdrone.pt")
    
    if not local_model_path.exists():
        try:
            model_path = hf_hub_download(
                repo_id="mshamrai/yolov8s-visdrone",
                filename="best.pt"
            )
            print(f"   Model downloaded to: {model_path}")
            
            # Copy to local directory
            import shutil
            shutil.copy(model_path, local_model_path)
            print(f"   Copied to: {local_model_path.absolute()}")
        except Exception as e:
            print(f"   ⚠️  Error downloading model: {e}")
            raise
    else:
        print(f"   Using existing model: {local_model_path.absolute()}")
    
    # Load the base model
    print(f"\n📥 Loading base model...")
    model = YOLO(str(local_model_path))
    
    # Start fine-tuning
    print(f"\n🔥 Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project="runs/finetune",
        name=output_name,
        exist_ok=True,
        
        # Fine-tuning specific settings
        patience=10,  # Early stopping patience
        save=True,    # Save checkpoints
        save_period=5,  # Save every 5 epochs
        
        # Augmentation (lighter for fine-tuning)
        hsv_h=0.01,   # Hue augmentation
        hsv_s=0.5,    # Saturation augmentation
        hsv_v=0.3,    # Value augmentation
        degrees=5.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.3,    # Scale
        flipud=0.0,   # No vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=0.5,   # Mosaic augmentation
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,    # Initial learning rate (lower for fine-tuning)
        lrf=0.01,     # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Results saved to: runs/finetune/{output_name}")
    
    # Get the best model path
    best_model_path = Path(f"runs/finetune/{output_name}/weights/best.pt")
    print(f"   Best model: {best_model_path}")
    
    # Validate the model
    print(f"\n📊 Validating best model...")
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    # Save a copy with a simple name
    output_path = Path(f"{output_name}.pt")
    import shutil
    shutil.copy(best_model_path, output_path)
    print(f"\n💾 Model saved to: {output_path.absolute()}")
    
    return best_model_path, metrics

if __name__ == "__main__":
    print("=" * 60)
    print("VisDrone Model Fine-tuning")
    print("=" * 60)
    
    # Run fine-tuning
    model_path, metrics = finetune_visdrone(
        base_model="mshamrai/yolov8s-visdrone",
        data_yaml="visdrone_finetuning/dataset.yaml",
        epochs=50,  # Adjust based on your needs
        batch_size=8,  # Adjust based on your GPU memory
        imgsz=640,
        output_name="visdrone_finetuned"
    )
    
    print("\n" + "=" * 60)
    print("🎉 Fine-tuning Complete!")
    print("=" * 60)
    print(f"\nYou can now use the fine-tuned model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('visdrone_finetuned.pt')")
    print(f"  results = model.predict('your_video.mp4')")
