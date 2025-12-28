"""
Process all 5 cameras using SAHI for superior aerial detection

SAHI (Slicing Aided Hyper Inference) dramatically improves detection:
- Standard YOLOv8: 2-6 detections per frame
- SAHI YOLOv8: 50-80+ detections per frame!
"""

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os

def process_video_with_sahi(video_path,
                            model_path='yolov8m.pt',
                            slice_height=320,
                            slice_width=320,
                            confidence=0.25,
                            save_every=50,
                            output_dir='output'):
    """
    Process video using SAHI sliced detection
    
    Args:
        video_path: Path to video file
        model_path: YOLO model to use (yolov8n/s/m/l/x)
        slice_height/width: Size of slices (smaller = more detections, slower)
        confidence: Confidence threshold
        save_every: Process every Nth frame
        output_dir: Where to save results
    """
    
    video_name = Path(video_path).stem
    print(f"\n{'='*70}")
    print(f"🎬 Processing: {video_name}")
    print(f"{'='*70}")
    
    # Load detection model
    print(f"\n📦 Loading model: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=confidence,
        device='cuda:0'
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Processing every {save_every} frames")
    
    frames_to_process = list(range(0, total_frames, save_every))
    print(f"   Will process: {len(frames_to_process)} frames")
    
    # Process frames
    detections = []
    temp_file = 'temp_frame.jpg'
    
    print(f"\n🔍 Running SAHI detection (slice size: {slice_width}x{slice_height})...")
    
    for frame_num in tqdm(frames_to_process, desc=f"   {video_name}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save frame temporarily
        cv2.imwrite(temp_file, frame)
        
        # Run SAHI sliced detection
        result = get_sliced_prediction(
            temp_file,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        
        # Extract detections
        for pred in result.object_prediction_list:
            detections.append({
                'frame': frame_num,
                'time_sec': frame_num / fps,
                'class': pred.category.name,
                'confidence': pred.score.value,
                'x1': int(pred.bbox.minx),
                'y1': int(pred.bbox.miny),
                'x2': int(pred.bbox.maxx),
                'y2': int(pred.bbox.maxy),
                'width': int(pred.bbox.maxx - pred.bbox.minx),
                'height': int(pred.bbox.maxy - pred.bbox.miny)
            })
    
    cap.release()
    
    # Cleanup temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Save results
    df = pd.DataFrame(detections)
    
    if len(df) > 0:
        output_path = Path(output_dir) / f'sahi_{video_name}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Statistics
        print(f"\n📊 Results:")
        print(f"   Total detections: {len(df)}")
        print(f"   Avg per frame: {len(df) / len(frames_to_process):.1f}")
        print(f"   Classes: {df['class'].value_counts().to_dict()}")
        print(f"   Saved to: {output_path}")
    else:
        print(f"\n⚠️  No detections found")
    
    return df

def process_all_cameras(video_dir='C:/Users/sakth/Documents/traffique_footage',
                       model='yolov8m.pt',
                       slice_size=320,
                       save_every=50):
    """
    Process all 5 camera videos with SAHI
    """
    print("="*70)
    print("🎥 MULTI-CAMERA SAHI PROCESSING")
    print("="*70)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {model}")
    print(f"   Slice size: {slice_size}x{slice_size}")
    print(f"   Save every: {save_every} frames")
    print(f"   Confidence: 0.25")
    
    # Find videos
    video_path = Path(video_dir)
    videos = sorted(video_path.glob('D*F*_stab.mp4'))
    
    if not videos:
        print(f"\n❌ No videos found in {video_dir}")
        return
    
    print(f"\n📹 Found {len(videos)} videos:")
    for v in videos:
        print(f"   - {v.name}")
    
    # Process each video
    all_results = {}
    
    for video in videos:
        df = process_video_with_sahi(
            video,
            model_path=model,
            slice_height=slice_size,
            slice_width=slice_size,
            save_every=save_every
        )
        all_results[video.stem] = df
    
    # Overall summary
    print("\n" + "="*70)
    print("📊 OVERALL SUMMARY")
    print("="*70)
    
    total_detections = sum(len(df) for df in all_results.values())
    total_frames = sum(len(df['frame'].unique()) if len(df) > 0 else 0 
                      for df in all_results.values())
    
    print(f"\n🎯 Total across all cameras:")
    print(f"   Detections: {total_detections}")
    print(f"   Frames processed: {total_frames}")
    print(f"   Avg per frame: {total_detections / total_frames:.1f}" if total_frames > 0 else "   No frames")
    
    print(f"\n📹 Per camera:")
    for video_name, df in all_results.items():
        if len(df) > 0:
            frames = len(df['frame'].unique())
            print(f"   {video_name}: {len(df)} detections ({len(df)/frames:.1f} per frame)")
        else:
            print(f"   {video_name}: 0 detections")
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Results saved in: output/sahi_*.csv")
    print(f"\n💡 Next steps:")
    print(f"   1. Review CSV files for detection quality")
    print(f"   2. Use detections for vehicle tracking")
    print(f"   3. Run analytics on the data")

if __name__ == '__main__':
    print("="*70)
    print("🔪 SAHI MULTI-CAMERA PROCESSOR")
    print("="*70)
    
    print("""
SAHI Results on Test Frame:
- Standard detection: 2-6 vehicles
- SAHI (320x320): 82 vehicles!
- SAHI (640x640): 52 vehicles

This is a 10-40x improvement with NO ANNOTATION NEEDED!
""")
    
    print("⚙️  Choose slice size:")
    print("   1. 320x320 - More detections, slower (~3-4x slower)")
    print("   2. 640x640 - Fewer detections, faster (~2x slower)")
    print("   3. Custom")
    
    choice = input("\nChoice (1/2/3) [default=2]: ").strip() or '2'
    
    if choice == '1':
        slice_size = 320
    elif choice == '3':
        slice_size = int(input("Enter slice size (e.g., 320, 480, 640): "))
    else:
        slice_size = 640
    
    print(f"\nHow often to process frames?")
    print(f"   1. Every frame (slowest, most complete)")
    print(f"   2. Every 25 frames (~1 second at 25fps)")
    print(f"   3. Every 50 frames (~2 seconds) - RECOMMENDED")
    print(f"   4. Every 100 frames (~4 seconds)")
    
    choice = input("\nChoice (1/2/3/4) [default=3]: ").strip() or '3'
    
    save_every_map = {'1': 1, '2': 25, '3': 50, '4': 100}
    save_every = save_every_map.get(choice, 50)
    
    print(f"\n🚀 Starting processing...")
    print(f"   Slice size: {slice_size}x{slice_size}")
    print(f"   Process every: {save_every} frames")
    
    # Estimate time
    print(f"\n⏱️  Estimated time per camera:")
    if slice_size == 320:
        print(f"   ~2-3 hours (135 slices per frame)")
    else:
        print(f"   ~1-1.5 hours (32 slices per frame)")
    print(f"   Total for 5 cameras: ~5-10 hours")
    print(f"   Can run overnight!")
    
    response = input("\nStart processing? (y/n): ").lower()
    
    if response == 'y':
        process_all_cameras(slice_size=slice_size, save_every=save_every)
    else:
        print("\n✅ Processing cancelled")
        print(f"   Run when ready: python process_multicam_sahi.py")
