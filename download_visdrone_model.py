"""
Download the VisDrone model from Hugging Face
"""
from huggingface_hub import hf_hub_download
from pathlib import Path

print("📥 Downloading VisDrone YOLOv8 model from Hugging Face...")

try:
    # Try to download the model
    model_path = hf_hub_download(
        repo_id="mshamrai/yolov8s-visdrone",
        filename="best.pt"
    )
    print(f"✅ Model downloaded successfully!")
    print(f"   Path: {model_path}")
    
    # Copy to local directory for easier access
    import shutil
    local_path = Path("yolov8s-visdrone.pt")
    shutil.copy(model_path, local_path)
    print(f"   Copied to: {local_path.absolute()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying to list available files in the repo...")
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files("mshamrai/yolov8s-visdrone")
        print(f"Available files:")
        for f in files:
            print(f"  - {f}")
    except Exception as e2:
        print(f"Could not list files: {e2}")
