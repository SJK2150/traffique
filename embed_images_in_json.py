#!/usr/bin/env python3
"""
Convert images to base64 and embed in Label Studio JSON
This is the most reliable way - no file serving needed!
"""

import json
import base64
from pathlib import Path
from tqdm import tqdm

print("🔄 Converting images to base64 embedded JSON...")

# Load existing JSON
with open('mydata/label_studio_tasks.json', 'r') as f:
    tasks = json.load(f)

# Process each task
images_dir = Path('mydata/images')
new_tasks = []

for task in tqdm(tasks, desc="Embedding images"):
    # Get image filename from path
    old_path = task['data']['image']
    filename = old_path.split('/')[-1]
    image_path = images_dir / filename
    
    if image_path.exists():
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Update task with base64 image
        task['data']['image'] = f"data:image/jpeg;base64,{img_data}"
        new_tasks.append(task)
    else:
        print(f"⚠️  Image not found: {image_path}")

# Save new JSON
output_file = 'label_studio_embedded.json'
with open(output_file, 'w') as f:
    json.dump(new_tasks, f, indent=2)

print(f"\n✅ Created: {output_file}")
print(f"   Tasks: {len(new_tasks)}")
print(f"   File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
print(f"\n📋 Next: Import {output_file} into Label Studio")
print(f"   No image files needed - everything is embedded!")
