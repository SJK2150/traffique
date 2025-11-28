"""
Utility script to setup project directories and check system requirements
"""

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version OK")
    return True


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
        
        return True
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return True


def create_directories():
    """Create project directory structure."""
    directories = [
        "data/videos",
        "data/maps",
        "data/calibration",
        "data/dataset/raw_frames",
        "data/dataset/images/train",
        "data/dataset/images/val",
        "data/dataset/images/test",
        "data/dataset/labels/train",
        "data/dataset/labels/val",
        "data/dataset/labels/test",
        "models",
        "output",
        "runs/train",
    ]
    
    print("\nCreating directory structure...")
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/

# Data
data/videos/*.mp4
data/videos/*.avi
data/dataset/
*.jpg
*.png
!data/maps/.gitkeep

# Models
models/*.pt
runs/

# Output
output/
*.csv

# IDE
.vscode/
.idea/
*.swp
"""
    
    gitignore_path = base_path / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✓ .gitignore created")
    
    # Create placeholder files
    placeholder_dirs = [
        "data/videos",
        "data/maps",
    ]
    
    for dir_name in placeholder_dirs:
        readme_path = base_path / dir_name / "README.txt"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"Place your files in this directory\n")


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required = [
        'cv2',
        'numpy',
        'pandas',
        'yaml',
        'ultralytics',
        'torch',
        'tqdm'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def download_base_model():
    """Download base YOLOv8 model."""
    print("\nChecking for base YOLOv8 model...")
    
    model_path = Path("yolov8n.pt")
    
    if model_path.exists():
        print("✓ Base model already exists")
        return True
    
    try:
        from ultralytics import YOLO
        print("Downloading yolov8n.pt...")
        model = YOLO('yolov8n.pt')
        print("✓ Base model downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def create_sample_config():
    """Create a sample config if none exists."""
    config_path = Path("config.yaml")
    
    if config_path.exists():
        print("\n✓ config.yaml already exists")
        return
    
    print("\n⚠ config.yaml not found - using default")
    print("  Please update paths in config.yaml before running")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("\n1. PREPARE DATA:")
    print("   - Place your drone videos in data/videos/")
    print("   - Get Google Earth screenshot and save to data/maps/global_map.jpg")
    
    print("\n2. EXTRACT FRAMES FOR LABELING:")
    print("   python extract_frames.py --video data/videos/your_video.mp4 --output data/dataset/raw_frames --fps 1 --max-frames 150")
    
    print("\n3. LABEL DATA:")
    print("   - Upload frames to Roboflow or CVAT")
    print("   - Label: Car, Bike, Pedestrian")
    print("   - Export in YOLOv8 format")
    
    print("\n4. TRAIN MODEL:")
    print("   python train_model.py prepare --dataset-dir data/dataset --source-images <path> --source-labels <path> --split")
    print("   python train_model.py train --dataset data/dataset/dataset.yaml --epochs 50")
    
    print("\n5. CALIBRATE CAMERAS:")
    print("   python calibration.py --camera-image data/videos/camera1_frame.jpg --map-image data/maps/global_map.jpg --output data/calibration/camera1_H.npy")
    
    print("\n6. UPDATE CONFIG:")
    print("   - Edit config.yaml with your paths")
    
    print("\n7. RUN SYSTEM:")
    print("   python main.py --config config.yaml")
    
    print("\n" + "="*60)
    print("For detailed instructions, see README.md")
    print("="*60 + "\n")


def main():
    """Main setup function."""
    print("="*60)
    print("Multi-Camera Traffic Analysis System - Setup")
    print("="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Check CUDA
        check_cuda()
        
        # Download base model
        download_base_model()
    
    # Check config
    create_sample_config()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
