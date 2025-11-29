"""
Quick test to compare standard YOLOv8 vs VisDrone-optimized YOLOv8
on a single frame with and without SAHI
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def test_models_comparison():
    """Compare standard vs VisDrone models with and without SAHI"""
    
    print("="*70)
    print("🔬 MODEL COMPARISON TEST")
    print("="*70)
    
    # Find a test frame
    video_dir = Path('C:/Users/sakth/Documents/traffique_footage')
    videos = list(video_dir.glob('D*F*_stab.mp4'))
    
    if not videos:
        print("❌ No videos found!")
        return
    
    # Extract a test frame
    video = videos[0]
    print(f"\n📹 Using video: {video.name}")
    
    cap = cv2.VideoCapture(str(video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get frame from middle of video
    test_frame_num = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame!")
        return
    
    # Save test frame
    test_image = 'test_frame_comparison.jpg'
    cv2.imwrite(test_image, frame)
    print(f"✅ Extracted frame {test_frame_num}")
    
    print("\n" + "="*70)
    print("TEST 1: Standard Detection (No SAHI)")
    print("="*70)
    
    # Test 1: Standard YOLOv8
    print("\n📦 Loading standard YOLOv8n...")
    standard_model = YOLO('yolov8n.pt')
    results = standard_model(test_image, verbose=False, conf=0.25)
    standard_count = len(results[0].boxes)
    print(f"   Detections: {standard_count}")
    
    # Test 2: VisDrone YOLOv8 - Download from HuggingFace
    print("\n📦 Downloading VisDrone YOLOv8s from HuggingFace...")
    print("   (This may take a few minutes on first run...)")
    try:
        from huggingface_hub import hf_hub_download
        visdrone_path = hf_hub_download(repo_id="mshamrai/yolov8s-visdrone", filename="best.pt")
        visdrone_model = YOLO(visdrone_path)
        results = visdrone_model(test_image, verbose=False, conf=0.25, max_det=1000)
        visdrone_count = len(results[0].boxes)
        print(f"   Detections: {visdrone_count}")
    except Exception as e:
        print(f"   ⚠️ Could not load VisDrone model: {e}")
        print(f"   Skipping VisDrone test...")
        visdrone_count = 0
        visdrone_path = None
    
    print("\n" + "="*70)
    print("TEST 2: With SAHI (640x640 slices)")
    print("="*70)
    
    # Test 3: Standard + SAHI
    print("\n📦 Standard YOLOv8n + SAHI...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov8n.pt',
        confidence_threshold=0.25,
        device='cuda:0'
    )
    result = get_sliced_prediction(
        test_image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )
    standard_sahi_count = len(result.object_prediction_list)
    print(f"   Detections: {standard_sahi_count}")
    
    # Test 4: VisDrone + SAHI
    if visdrone_path:
        print("\n📦 VisDrone YOLOv8s + SAHI...")
        detection_model_vd = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=visdrone_path,
            confidence_threshold=0.25,
            device='cuda:0'
        )
        result = get_sliced_prediction(
            test_image,
            detection_model_vd,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        visdrone_sahi_count = len(result.object_prediction_list)
        print(f"   Detections: {visdrone_sahi_count}")
    else:
        visdrone_sahi_count = 0
    
    # Summary
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<30} {'Detections':<15} {'Improvement'}")
    print("-" * 70)
    print(f"{'1. Standard YOLOv8n':<30} {standard_count:<15} {'Baseline'}")
    if visdrone_count > 0:
        print(f"{'2. VisDrone YOLOv8s':<30} {visdrone_count:<15} {f'+{visdrone_count-standard_count} ({visdrone_count/standard_count:.1f}x)' if standard_count > 0 else 'N/A'}")
    print(f"{'3. Standard + SAHI (640)':<30} {standard_sahi_count:<15} {f'+{standard_sahi_count-standard_count} ({standard_sahi_count/standard_count:.1f}x)' if standard_count > 0 else 'N/A'}")
    if visdrone_sahi_count > 0:
        print(f"{'4. VisDrone + SAHI (640)':<30} {visdrone_sahi_count:<15} {f'+{visdrone_sahi_count-standard_count} ({visdrone_sahi_count/standard_count:.1f}x)' if standard_count > 0 else 'N/A'}")
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)
    
    best_method = max(
        [('Standard', standard_count), 
         ('VisDrone', visdrone_count),
         ('Standard + SAHI', standard_sahi_count),
         ('VisDrone + SAHI', visdrone_sahi_count)],
        key=lambda x: x[1]
    )
    
    print(f"\n✨ Best method: {best_method[0]} with {best_method[1]} detections")
    
    if visdrone_sahi_count > standard_sahi_count and visdrone_sahi_count > 0:
        improvement = ((visdrone_sahi_count - standard_sahi_count) / standard_sahi_count * 100)
        print(f"\n🚀 VisDrone + SAHI gives {improvement:.1f}% more detections!")
        print(f"   Use: python process_multicam_sahi_visdrone.py")
    else:
        print(f"\n💡 Standard SAHI is sufficient for your footage")
        print(f"   Use: python process_multicam_sahi.py")
    
    # Cleanup
    import os
    if os.path.exists('temp_visdrone.pt'):
        os.remove('temp_visdrone.pt')
    
    print("\n✅ Test complete!")

if __name__ == '__main__':
    test_models_comparison()
