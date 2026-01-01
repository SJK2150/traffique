#!/usr/bin/env python3
"""
Prepare Data for Label Studio Fine-tuning

1. Extracts frames from video (every Nth frame to avoid redundancy)
2. Runs VisDrone model to generate pre-annotations
3. Exports in Label Studio JSON format
4. Ready to import into Label Studio for manual correction
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
import argparse
from huggingface_hub import hf_hub_download


class LabelStudioPreparation:
    """Prepare video frames and annotations for Label Studio"""
    
    def __init__(self, video_path, output_dir="mydata", confidence=0.25):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.confidence = confidence
        
        # Create directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # VisDrone classes
        self.visdrone_classes = {
            0: 'Pedestrian',
            1: 'People',
            2: 'Bicycle',
            3: 'Car',
            4: 'Van',
            5: 'Truck',
            6: 'Tricycle',
            7: 'Awning-tricycle',
            8: 'Bus',
            9: 'Motor'
        }
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖼️  Images will be saved to: {self.images_dir}")
    
    def load_model(self):
        """Load VisDrone model"""
        print("\n📥 Loading VisDrone model...")
        try:
            model_path = hf_hub_download(
                repo_id="mshamrai/yolov8s-visdrone",
                filename="best.pt"
            )
            print("   ✓ Downloaded from HuggingFace")
        except Exception as e:
            print(f"   ⚠️  Using local model")
            model_path = "best.pt"
        
        self.model = YOLO(model_path)
        print("   ✓ Model loaded")
    
    def extract_and_annotate(self, start_frame=0, num_frames=100, frame_skip=10):
        """
        Extract frames and generate annotations
        
        Args:
            start_frame: Starting frame number
            num_frames: Total frames to process
            frame_skip: Extract every Nth frame (e.g., 10 = every 10th frame)
        """
        print(f"\n🎬 Processing video: {Path(self.video_path).name}")
        print(f"   Start frame: {start_frame}")
        print(f"   Frames to process: {num_frames}")
        print(f"   Frame skip: {frame_skip} (extracting every {frame_skip}th frame)")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Video: {width}x{height} @ {fps} FPS")
        print(f"   Total frames in video: {total_frames}")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Storage for Label Studio format
        label_studio_tasks = []
        
        frame_num = start_frame
        extracted_count = 0
        
        pbar = tqdm(total=num_frames, desc="Extracting & Annotating")
        
        while frame_num < start_frame + num_frames and frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if (frame_num - start_frame) % frame_skip == 0:
                # Save frame
                image_filename = f"frame_{frame_num:06d}.jpg"
                image_path = self.images_dir / image_filename
                cv2.imwrite(str(image_path), frame)
                
                # Run detection
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Convert to Label Studio format
                annotations = []
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        if cls in self.visdrone_classes:
                            # Convert to Label Studio format (percentage coordinates)
                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            
                            annotation = {
                                "original_width": width,
                                "original_height": height,
                                "image_rotation": 0,
                                "value": {
                                    "x": float((x1 / width) * 100),
                                    "y": float((y1 / height) * 100),
                                    "width": float(((x2 - x1) / width) * 100),
                                    "height": float(((y2 - y1) / height) * 100),
                                    "rotation": 0,
                                    "rectanglelabels": [self.visdrone_classes[cls]]
                                },
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "score": float(conf)
                            }
                            annotations.append(annotation)
                
                # Create Label Studio task
                task = {
                    "data": {
                        "image": f"/data/local-files/?d=mydata/images/{image_filename}"
                    },
                    "predictions": [{
                        "result": annotations,
                        "score": 1.0,
                        "model_version": "visdrone-yolov8s"
                    }]
                }
                
                label_studio_tasks.append(task)
                extracted_count += 1
            
            frame_num += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Save Label Studio JSON
        output_json = self.output_dir / "label_studio_tasks.json"
        with open(output_json, 'w') as f:
            json.dump(label_studio_tasks, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"   Extracted frames: {extracted_count}")
        print(f"   Images saved to: {self.images_dir}")
        print(f"   Annotations saved to: {output_json}")
        
        print(f"\n📋 Next steps:")
        print(f"   1. In Label Studio, go to your project")
        print(f"   2. Click 'Import' → 'Upload Files'")
        print(f"   3. Upload: {output_json}")
        print(f"   4. Set 'Treat CSV/TSV as' → 'List of tasks'")
        print(f"   5. Start annotating and correcting predictions!")
        
        return output_json


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Label Studio fine-tuning")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", default="label_studio_data", help="Output directory")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame")
    parser.add_argument("--num-frames", type=int, default=500, help="Number of frames to process")
    parser.add_argument("--frame-skip", type=int, default=10, 
                       help="Extract every Nth frame (default: 10)")
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    print("="*70)
    print("  LABEL STUDIO DATA PREPARATION")
    print("  Extract Frames + Generate Pre-annotations")
    print("="*70)
    
    prep = LabelStudioPreparation(
        args.video,
        output_dir=args.output_dir,
        confidence=args.confidence
    )
    
    prep.load_model()
    prep.extract_and_annotate(
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        frame_skip=args.frame_skip
    )
    
    print("\n✅ Ready for Label Studio!")


if __name__ == "__main__":
    main()
