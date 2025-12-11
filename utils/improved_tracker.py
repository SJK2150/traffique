"""
Improved Kalman-based tracker for better motion prediction.
Replaces simple greedy matching with state estimation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class KalmanFilterTrack:
    """1D Kalman filter for bbox coordinate tracking"""
    
    def __init__(self, z0, dt=1.0, process_var=1.0, meas_var=10.0):
        """
        Args:
            z0: Initial measurement (position)
            dt: Time step
            process_var: Process noise variance (how much we expect motion to change)
            meas_var: Measurement noise variance (detector uncertainty)
        """
        # State: [position, velocity]
        self.x = np.array([float(z0), 0.0])  # pos, vel
        self.v = 0.0  # velocity component
        
        # State transition
        self.dt = float(dt)
        
        # Covariance
        self.p = 100.0  # Uncertainty in position
        self.q = float(process_var)  # Process noise
        self.r = float(meas_var)  # Measurement noise
    
    def predict(self):
        """Predict next state"""
        # Simple velocity model: x' = x + v*dt
        self.x[0] = self.x[0] + self.x[1] * self.dt
        self.p = self.p + self.q  # Increase uncertainty
        return float(self.x[0])
    
    def update(self, z):
        """Update with measurement"""
        z = float(z)
        # Kalman gain (simplified)
        k = self.p / (self.p + self.r)
        
        # Update position estimate
        innovation = z - self.x[0]
        self.x[0] = self.x[0] + k * innovation
        self.x[1] = self.x[1] + 0.5 * k * innovation  # Simple velocity update
        
        # Update uncertainty
        self.p = (1.0 - k) * self.p
        
        return float(self.x[0])


class KalmanTrack:
    """Track with Kalman filtering for each bbox coordinate"""
    
    def __init__(self, track_id: int, init_bbox: Tuple[int, int, int, int], 
                 frame_idx: int, embedding: Optional[np.ndarray] = None, dt: float = 1.0):
        self.id = track_id
        self.bboxes = [init_bbox]
        self.frames = [frame_idx]
        self.embedding = embedding.copy() if embedding is not None else None
        self.missed = 0
        
        # Kalman filters for each coordinate
        x1, y1, x2, y2 = init_bbox
        self.kf_x1 = KalmanFilterTrack(x1, dt=dt, process_var=5.0, meas_var=10.0)
        self.kf_y1 = KalmanFilterTrack(y1, dt=dt, process_var=5.0, meas_var=10.0)
        self.kf_x2 = KalmanFilterTrack(x2, dt=dt, process_var=5.0, meas_var=10.0)
        self.kf_y2 = KalmanFilterTrack(y2, dt=dt, process_var=5.0, meas_var=10.0)
    
    def last_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bboxes[-1]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def predicted_center(self) -> Tuple[float, float]:
        """Get predicted center based on Kalman state"""
        x1_pred = self.kf_x1.predict()
        y1_pred = self.kf_y1.predict()
        x2_pred = self.kf_x2.predict()
        y2_pred = self.kf_y2.predict()
        return ((x1_pred + x2_pred) / 2.0, (y1_pred + y2_pred) / 2.0)
    
    def update(self, bbox: Tuple[int, int, int, int], frame_idx: int, 
               embedding: Optional[np.ndarray] = None, ema_alpha: float = 0.2):
        """Update track with new detection"""
        x1, y1, x2, y2 = bbox
        
        # Update Kalman filters
        self.kf_x1.update(x1)
        self.kf_y1.update(y1)
        self.kf_x2.update(x2)
        self.kf_y2.update(y2)
        
        self.bboxes.append(bbox)
        self.frames.append(frame_idx)
        self.missed = 0
        
        # Update embedding
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                self.embedding = (1 - ema_alpha) * self.embedding + ema_alpha * embedding
    
    def predict_next(self) -> Tuple[int, int, int, int]:
        """Predict next bbox (for filling gaps)"""
        x1_pred = int(self.kf_x1.predict())
        y1_pred = int(self.kf_y1.predict())
        x2_pred = int(self.kf_x2.predict())
        y2_pred = int(self.kf_y2.predict())
        return (x1_pred, y1_pred, x2_pred, y2_pred)


class ImprovedOnlineTracker:
    """
    Improved online tracker with:
    - Kalman filtering for motion prediction
    - Better ID stability
    - Gap filling with predictions
    """
    
    def __init__(self, max_missed: int = 30, dist_thresh_px: float = 80.0, 
                 appearance_weight: float = 0.6, ema_alpha: float = 0.2):
        self.dist_thresh = dist_thresh_px
        self.max_missed = max_missed
        self.ema_alpha = ema_alpha
        self.appearance_weight = float(appearance_weight)
        self.next_id = 1
        self.tracks: List[KalmanTrack] = []
        self.dt = 1.0  # Time step (1 frame)
    
    def _center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def update(self, detections: List[Dict[str, Any]], frame_idx: int):
        """Update tracks with detections using Hungarian-like assignment"""
        if len(detections) == 0:
            # Age all tracks
            for t in self.tracks:
                t.missed += 1
            return
        
        det_centers = [self._center(d['bbox']) for d in detections]
        det_bboxes = [d['bbox'] for d in detections]
        det_embs = [d.get('embedding') for d in detections]
        
        # Predict and compute costs
        costs = np.full((len(self.tracks), len(detections)), np.inf)
        
        for i, track in enumerate(self.tracks):
            if track.missed >= self.max_missed:
                continue  # Don't assign to dead tracks
            
            pred_center = track.predicted_center()
            
            for j, det_center in enumerate(det_centers):
                # Motion cost
                dx = float(pred_center[0]) - float(det_center[0])
                dy = float(pred_center[1]) - float(det_center[1])
                pd2 = dx * dx + dy * dy
                
                if pd2 > (self.dist_thresh ** 2):
                    continue  # Gating: too far
                
                motion_cost = np.sqrt(pd2) / self.dist_thresh
                
                # Appearance cost
                app_cost = 0.0
                te = track.embedding
                de = det_embs[j]
                if te is not None and de is not None and te.size == de.size:
                    te_norm = np.linalg.norm(te)
                    de_norm = np.linalg.norm(de)
                    if te_norm > 1e-6 and de_norm > 1e-6:
                        app_cost = 1.0 - float(np.dot(te.ravel() / te_norm, de.ravel() / de_norm))
                        app_cost = max(0.0, min(2.0, app_cost))
                
                # Combined cost
                cost = (1.0 - self.appearance_weight) * motion_cost + self.appearance_weight * app_cost
                costs[i, j] = cost
        
        # Greedy assignment (simple, but works for most cases)
        assigned_dets = set()
        for i in range(len(self.tracks)):
            if self.tracks[i].missed >= self.max_missed:
                continue
            
            j = np.argmin(costs[i])
            if costs[i, j] < np.inf and j not in assigned_dets:
                self.tracks[i].update(det_bboxes[j], frame_idx, det_embs[j], self.ema_alpha)
                assigned_dets.add(j)
            else:
                self.tracks[i].missed += 1
        
        # Create new tracks for unassigned detections
        for j in range(len(detections)):
            if j not in assigned_dets:
                new_track = KalmanTrack(self.next_id, det_bboxes[j], frame_idx, det_embs[j], self.dt)
                self.tracks.append(new_track)
                self.next_id += 1
    
    def get_active_tracks(self, min_length: int = 1) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Get active tracks as dict of track_id -> list of bboxes"""
        active = {}
        for track in self.tracks:
            if track.missed < self.max_missed and len(track.bboxes) >= min_length:
                active[track.id] = track.bboxes
        return active
    
    def get_all_tracks(self) -> List[KalmanTrack]:
        """Get all track objects"""
        return self.tracks
