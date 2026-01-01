#!/usr/bin/env python3
"""
Create Label Studio import with base64 encoded images
This avoids all local file serving issues
"""

import cv2
import json
import base64
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_label_studio_json(video_path, output_json, start_frame=0, num_frames=500, 
                             frame_skip=25, confidence=0.20):
    """Create Label Studio JSON with embedded base64 images"""
    
    print("="*70)
    print("  LABEL STUDIO - BASE64 EMBEDDED IMAGES")
    print("  No local file serving needed!")
    print("="*70)
    
    # Load model
    print("\n📥 Loading VisDrone model...")
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
    
    # VisDrone classes
    visdrone_classes = {
        0: 'Pedestrian', 1: 'People', 2: 'Bicycle', 3: 'Car',
        4: 'Van', 5: 'Truck', 6: 'Tricycle', 7: 'Awning-tricycle',
        8: 'Bus', 9: 'Motor'
    }
    
    print(f"\n🎬 Processing video: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    tasks = []
    frame_num = start_frame
    
    pbar = tqdm(total=num_frames, desc="Processing")
    
    while frame_num < start_frame + num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_num - start_frame) % frame_skip == 0:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Run detection
            results = model.predict(
                frame, conf=confidence, verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Build annotations
            annotations = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls in visdrone_classes:
                        x1, y1, x2, y2 = map(float, xyxy)
                        
                        annotations.append({
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "x": float((x1 / width) * 100),
                                "y": float((y1 / height) * 100),
                                "width": float(((x2 - x1) / width) * 100),
                                "height": float(((y2 - y1) / height) * 100),
                                "rotation": 0,
                                "rectanglelabels": [visdrone_classes[cls]]
                            },
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "score": float(conf)
                        })
            
            # Create task
            task = {
                "data": {
                    "image": f"data:image/jpeg;base64,{img_base64}"
                },
                "predictions": [{
                    "result": annotations,
                    "score": 1.0
                }]
            }
            tasks.append(task)
        
        frame_num += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"\n✅ Created: {output_json}")
    print(f"   Tasks: {len(tasks)}")
    print(f"\n📋 Next: Import this JSON into Label Studio")
    print(f"   No image upload needed - images are embedded!")


if __name__ == "__main__":
    create_label_studio_json(
        video_path=r"C:\Users\sakth\Documents\traffique_footage\D2F1_stab.mp4",
        output_json="label_studio_embedded.json",
        start_frame=0,
        num_frames=500,
        frame_skip=25,
        confidence=0.20
    )
