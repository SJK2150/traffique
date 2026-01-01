import cv2
import numpy as np
from ultralytics import YOLO

class HorizontalSplitDetector:
    def __init__(self, model_path="visdrone_finetuned.pt", split_ratio=0.5, overlap=0.1):
        """
        Detector that splits image horizontally to better detect small far objects.
        """
        self.model = YOLO(model_path)
        self.split_ratio = split_ratio  # Where to cut (0.5 = middle)
        self.overlap = overlap
        
    def detect(self, frame):
        height, width = frame.shape[:2]
        split_y = int(height * self.split_ratio)
        overlap_px = int(height * self.overlap)
        
        # 1. Bottom ROI (Near field - usually easy)
        # Start a bit higher to cover overlap
        y1_btm = max(0, split_y - overlap_px)
        roi_bottom = frame[y1_btm:, :]
        
        # 2. Top ROI (Far field - hard, small objects)
        # End a bit lower to cover overlap
        y2_top = min(height, split_y + overlap_px)
        roi_top = frame[:y2_top, :]
        
        # Run inference
        # We can upscale the top ROI to help with small objects if needed
        # roi_top_zoomed = cv2.resize(roi_top, None, fx=1.5, fy=1.5)
        
        results_top = self.model.predict(roi_top, conf=0.25, verbose=False)[0]
        results_btm = self.model.predict(roi_bottom, conf=0.25, verbose=False)[0]
        
        detections = []
        
        # Process Top Results
        for box in results_top.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            # No offset needed for y, top starts at 0
            detections.append([x1, y1, x2, y2, conf, cls])
            
        # Process Bottom Results
        for box in results_btm.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            # Add offset to y
            detections.append([x1, y1 + y1_btm, x2, y2 + y1_btm, conf, cls])
            
        # NMS to merge overlapping boxes at the seam
        if len(detections) > 0:
             return self.nms(detections)
        return []

    def nms(self, dets, thresh=0.5):
        # Simple NMS or just use a library function
        # For simplicity returning raw list for now or using cv2.dnn.NMSBoxes if needed
        # But Ultralytics usually does NMS per inference. We only need to handle the seam.
        # A simple box merge is sufficient.
        return dets

if __name__ == "__main__":
    # Test on video
    detector = HorizontalSplitDetector()
    cap = cv2.VideoCapture("C:/Users/sakth/Documents/traffique_footage/D2F1_stab.mp4")
    out = cv2.VideoWriter("split_detect_test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    
    for _ in range(100):
        ret, frame = cap.read()
        if not ret: break
        dets = detector.detect(frame)
        
        for d in dets:
            x1, y1, x2, y2, _, cls = map(int, d[:6])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        out.write(frame)
    cap.release()
    out.release()
