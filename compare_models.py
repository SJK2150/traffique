"""
Test different YOLO models to find the best one for drone footage
Compares YOLOv8 Nano vs Small vs Medium vs Large
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time

def compare_models(test_video='C:/Users/sakth/Documents/traffique_footage/D5F1_stab.mp4',
                   test_frame=232):
    """
    Compare different YOLO model sizes on the same frame
    """
    
    print("\n" + "="*70)
    print("🔬 COMPARING YOLO MODEL SIZES")
    print("="*70)
    
    # Models to test (from smallest to largest)
    models = {
        'yolov8n.pt': 'Nano (3MB - fastest, least accurate)',
        'yolov8s.pt': 'Small (11MB - balanced)',
        'yolov8m.pt': 'Medium (26MB - better accuracy)',
        'yolov8l.pt': 'Large (44MB - high accuracy)',
        'yolov8x.pt': 'XLarge (68MB - best accuracy, slowest)'
    }
    
    # Load test frame
    print(f"\n📹 Loading frame {test_frame} from video...")
    cap = cv2.VideoCapture(test_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Error: Could not load frame {test_frame}")
        return
    
    print("✅ Frame loaded\n")
    
    results = {}
    
    # Test each model
    for model_name, description in models.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} - {description}")
        print('='*70)
        
        try:
            # Load model
            print(f"  📦 Loading model...")
            model = YOLO(model_name)
            
            # Run inference
            print(f"  🔍 Running detection...")
            start_time = time.time()
            detections = model(frame, conf=0.25, verbose=False)[0]
            inference_time = time.time() - start_time
            
            boxes = detections.boxes
            
            # Count by class
            class_counts = {}
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            results[model_name] = {
                'total': len(boxes),
                'classes': class_counts,
                'time': inference_time,
                'boxes': boxes,
                'model': model
            }
            
            print(f"\n  📊 Results:")
            print(f"     Total detections: {len(boxes)}")
            print(f"     Inference time: {inference_time:.3f}s")
            print(f"     Classes: {class_counts}")
            
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
            print(f"     (Model will auto-download on first use)")
            results[model_name] = None
    
    # Print comparison
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Detections':<12} {'Time (s)':<12} {'Speed (FPS)':<12}")
    print("-"*70)
    
    for model_name, result in results.items():
        if result:
            detections = result['total']
            time_taken = result['time']
            fps = 1.0 / time_taken if time_taken > 0 else 0
            print(f"{model_name:<15} {detections:<12} {time_taken:<12.3f} {fps:<12.1f}")
    
    print("="*70)
    
    # Show best model visually
    best_model = max([k for k, v in results.items() if v], 
                     key=lambda k: results[k]['total'])
    print(f"\n🏆 Best model: {best_model}")
    print(f"   Detected {results[best_model]['total']} vehicles")
    print(f"   vs {results['yolov8n.pt']['total']} with current Nano model")
    print(f"   Improvement: +{results[best_model]['total'] - results['yolov8n.pt']['total']} vehicles")
    
    # Draw comparison
    print(f"\n👁️  Showing visual comparison...")
    
    # Create comparison frames
    comparison_frames = []
    for model_name in ['yolov8n.pt', best_model]:
        if results[model_name]:
            frame_copy = frame.copy()
            boxes = results[model_name]['boxes']
            model_obj = results[model_name]['model']
            
            # Draw boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model_obj.names[cls]
                
                color = (0, 255, 0) if class_name == 'car' else (255, 0, 0)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, f"{class_name} {conf:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add title
            title = f"{model_name}: {len(boxes)} vehicles"
            cv2.putText(frame_copy, title, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            comparison_frames.append(frame_copy)
    
    # Stack and show
    if len(comparison_frames) == 2:
        comparison = cv2.hconcat(comparison_frames)
        output_path = 'output/model_comparison.jpg'
        cv2.imwrite(output_path, comparison)
        print(f"💾 Saved to: {output_path}")
        
        cv2.namedWindow('Model Comparison', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Model Comparison', 1920, 540)
        cv2.imshow('Model Comparison', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    print(f"Use: {best_model}")
    print(f"Reason: Best detection ({results[best_model]['total']} vehicles)")
    print(f"Speed: {1.0/results[best_model]['time']:.1f} FPS")
    print(f"\nTo use this model, update process_multicam.py:")
    print(f"  model = YOLO('{best_model}')")
    print("="*70)
    
    return best_model, results

if __name__ == '__main__':
    compare_models()
