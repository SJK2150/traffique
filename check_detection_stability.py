from ultralytics import YOLO
import cv2
import numpy as np

def check_stability(
    video_path="C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4",
    model_path="visdrone_finetuned.pt",
    output_path="detection_stability_check.avi",
    frames=300
):
    print(f"🔬 Checking detection stability on {video_path}...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Video writer (MJPG .avi)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    
    frame_count = 0
    while frame_count < frames:
        ret, frame = cap.read()
        if not ret: break
        
        # Run inference
        results = model.predict(frame, conf=0.15, verbose=False)[0] # Low conf to see "weak" detections
        
        # Plot RAW boxes
        # We manually draw them to ensure we see exactly what the model sees
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Color code by confidence
            # High conf (>0.5) = Green
            # Low conf (<0.3) = Red (New model is unsure)
            if conf > 0.5: color = (0, 255, 0)
            elif conf > 0.3: color = (0, 255, 255)
            else: color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Green: Conf>0.5 | Yellow: Conf>0.3 | Red: Unsure", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    check_stability()
