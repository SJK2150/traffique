"""
Use SAHI (Slicing Aided Hyper Inference) for aerial detection

SAHI slices large aerial images into smaller pieces, runs detection on each,
then merges results. This dramatically improves small object detection!

Perfect for drone footage where vehicles appear small.
"""

import cv2
from pathlib import Path

def install_sahi():
    """Install SAHI package"""
    import subprocess
    import sys
    
    print("\n📦 Installing SAHI...")
    try:
        import sahi
        print("✅ SAHI already installed")
        return True
    except ImportError:
        print("   Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sahi", "-q"])
        print("✅ Installed!")
        return True

def test_sahi_detection(video_path='C:/Users/sakth/Documents/traffique_footage/D5F1_stab.mp4',
                        test_frame=232):
    """
    Test SAHI with YOLOv8 on aerial footage
    """
    print("\n" + "="*70)
    print("🔪 TESTING SAHI (SLICED DETECTION)")
    print("="*70)
    
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO
    
    # Load frame
    print(f"\n📹 Loading frame {test_frame}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not load frame")
        return
    
    h, w = frame.shape[:2]
    print(f"   Frame size: {w}x{h}")
    
    # Save frame temporarily
    temp_path = 'temp_frame.jpg'
    cv2.imwrite(temp_path, frame)
    
    # Test different models
    models_to_test = [
        ('YOLOv8-Nano', 'yolov8n.pt'),
        ('YOLOv8-Small', 'yolov8s.pt'),
        ('YOLOv8-Medium', 'yolov8m.pt'),
    ]
    
    results = {}
    
    for name, model_path in models_to_test:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {name} with SAHI")
        print(f"{'='*70}")
        
        # Standard detection (no slicing)
        print(f"\n1️⃣  Standard detection (no slicing)...")
        standard_model = YOLO(model_path)
        standard_results = standard_model(frame, conf=0.25, verbose=False)[0]
        standard_count = len(standard_results.boxes)
        print(f"   Detections: {standard_count}")
        
        # SAHI detection (with slicing)
        print(f"\n2️⃣  SAHI sliced detection...")
        
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.25,
            device='cuda:0'
        )
        
        # Try different slice sizes
        slice_results = []
        for slice_h, slice_w in [(320, 320), (640, 640)]:
            print(f"\n   Slice size: {slice_w}x{slice_h}")
            
            result = get_sliced_prediction(
                temp_path,
                detection_model,
                slice_height=slice_h,
                slice_width=slice_w,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            sahi_count = len(result.object_prediction_list)
            print(f"   Detections: {sahi_count}")
            
            slice_results.append({
                'slice_size': f'{slice_w}x{slice_h}',
                'count': sahi_count,
                'result': result
            })
        
        results[name] = {
            'standard': standard_count,
            'sliced': slice_results,
            'model': standard_model
        }
    
    # Cleanup
    Path(temp_path).unlink()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"   Standard: {data['standard']} detections")
        for slice_data in data['sliced']:
            print(f"   SAHI ({slice_data['slice_size']}): {slice_data['count']} detections")
    
    # Find best
    best_improvement = 0
    best_config = None
    
    for name, data in results.items():
        for slice_data in data['sliced']:
            improvement = slice_data['count'] - data['standard']
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = (name, slice_data['slice_size'], slice_data['count'])
    
    if best_config:
        print(f"\n🏆 BEST RESULT:")
        print(f"   Model: {best_config[0]}")
        print(f"   Slice: {best_config[1]}")
        print(f"   Detections: {best_config[2]}")
        print(f"   Improvement: +{best_improvement} vehicles vs standard!")
        
        if best_config[2] >= 15:
            print(f"\n✅ EXCELLENT! {best_config[2]} vehicles detected!")
            print(f"   This is much better than baseline (6 vehicles)")
            print(f"\n🎯 Use SAHI for all processing - NO ANNOTATION NEEDED!")
        elif best_config[2] > 10:
            print(f"\n⚠️ GOOD: {best_config[2]} vehicles (better than baseline)")
            print(f"   Consider using SAHI OR fine-tuning with manual data")
        else:
            print(f"\n⚠️ Only {best_config[2]} vehicles detected")
            print(f"   Manual annotation may still be needed")
    
    return results

def create_sahi_processor():
    """
    Create a script to process all videos with SAHI
    """
    print("\n" + "="*70)
    print("📝 CREATING SAHI PROCESSOR")
    print("="*70)
    
    script_content = """'''
Process multi-camera footage using SAHI for better small object detection
'''

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def process_with_sahi(video_path, model_path='yolov8m.pt', 
                      slice_height=640, slice_width=640,
                      conf=0.25, save_every=50):
    '''Process video with SAHI sliced detection'''
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf,
        device='cuda:0'
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections = []
    
    for frame_num in tqdm(range(0, total_frames, save_every)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame temporarily
        cv2.imwrite('temp.jpg', frame)
        
        # Run SAHI
        result = get_sliced_prediction(
            'temp.jpg',
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        
        # Extract detections
        for pred in result.object_prediction_list:
            detections.append({
                'frame': frame_num,
                'class': pred.category.name,
                'confidence': pred.score.value,
                'x1': pred.bbox.minx,
                'y1': pred.bbox.miny,
                'x2': pred.bbox.maxx,
                'y2': pred.bbox.maxy
            })
    
    cap.release()
    Path('temp.jpg').unlink()
    
    return pd.DataFrame(detections)

if __name__ == '__main__':
    # Process all 5 cameras
    video_dir = Path('C:/Users/sakth/Documents/traffique_footage')
    videos = sorted(video_dir.glob('D*F*_stab.mp4'))
    
    for video in videos:
        print(f'\\nProcessing {video.name}...')
        df = process_with_sahi(str(video))
        output = f'output/sahi_{video.stem}.csv'
        df.to_csv(output, index=False)
        print(f'Saved: {output}')
        print(f'Total detections: {len(df)}')
'''
    
    with open('process_multicam_sahi.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created: process_multicam_sahi.py")
    print("\n   Use this to process all videos with SAHI!")

if __name__ == '__main__':
    print("="*70)
    print("🔪 SAHI - SLICED DETECTION FOR AERIAL IMAGERY")
    print("="*70)
    
    print("""
SAHI (Slicing Aided Hyper Inference) is designed for aerial imagery!

How it works:
1. Slices large image into smaller overlapping pieces
2. Runs YOLO on each piece (easier to detect small objects)
3. Merges results with overlap handling

Benefits:
- Better detection of small/distant vehicles
- Works with standard YOLO (no custom training needed!)
- Can improve detections by 2-3x
- NO ANNOTATION REQUIRED!

Let's test it on your footage...
""")
    
    response = input("\nTry SAHI? (y/n): ").lower()
    
    if response == 'y':
        # Install SAHI
        if install_sahi():
            # Test it
            results = test_sahi_detection()
            
            # Create processor
            create_sahi_processor()
            
            print("\n" + "="*70)
            print("✅ DONE!")
            print("="*70)
            print("\nNext steps:")
            print("1. If SAHI worked well: Run process_multicam_sahi.py")
            print("2. If not: Go back to manual annotation")
    else:
        print("\n✅ Sticking with other approaches")
