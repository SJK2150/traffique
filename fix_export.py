#!/usr/bin/env python3
"""
Extract images from Label Studio export and match with labels
"""

import json
import base64
from pathlib import Path
import shutil

print("🔍 Fixing Label Studio export...")

# Check what we have
export_dir = Path("label_studio_export")
labels_dir = export_dir / "labels"
images_dir = export_dir / "images"

# Create images directory
images_dir.mkdir(exist_ok=True)

# Find label files
label_files = list(labels_dir.glob("*.txt"))
print(f"   Found {len(label_files)} label files")

# Get original images from mydata
source_images = Path("mydata/images")
if source_images.exists():
    print(f"\n📁 Copying images from mydata/images...")
    
    # Copy all images
    for img_file in source_images.glob("*.jpg"):
        dest = images_dir / img_file.name
        shutil.copy(img_file, dest)
        print(f"   ✓ {img_file.name}")
    
    copied_images = list(images_dir.glob("*.jpg"))
    print(f"\n✅ Copied {len(copied_images)} images")
    
    # Now re-run prepare_training_data
    print("\n🔄 Re-running training data preparation...")
    import subprocess
    result = subprocess.run([
        "python", "prepare_training_data.py",
        "--export-dir", "label_studio_export",
        "--output-dir", "training_data"
    ])
    
    if result.returncode == 0:
        print("\n✅ Training data ready!")
    else:
        print("\n⚠️  Error preparing training data")
else:
    print(f"\n⚠️  Source images not found at {source_images}")
    print("   Please check if mydata/images/ exists")
