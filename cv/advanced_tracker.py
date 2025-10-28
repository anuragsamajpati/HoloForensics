import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection.get('class_id', 0)
        self.class_name = detection.get('class_name', 'unknown')
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        
    def update(self, detection):
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        
    def predict(self):
        self.time_since_update += 1

class AdvancedTracker:
    def __init__(self, max_disappeared=10, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
    def calculate_iou(self, box1, box2):
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def update(self, detections):
        # Predict existing tracks
        for track in self.tracks.values():
            track.predict()
            
        # Calculate cost matrix
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            track_ids = list(self.tracks.keys())
            
            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for j, detection in enumerate(detections):
                    iou = self.calculate_iou(track.bbox, detection['bbox'])
                    cost_matrix[i, j] = 1.0 - iou  # Convert to cost
                    
            # Hungarian algorithm for assignment
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            matched_tracks = set()
            matched_detections = set()
            
            for track_idx, det_idx in zip(track_indices, det_indices):
                if cost_matrix[track_idx, det_idx] < (1.0 - self.iou_threshold):
                    track_id = track_ids[track_idx]
                    self.tracks[track_id].update(detections[det_idx])
                    matched_tracks.add(track_id)
                    matched_detections.add(det_idx)
            
            # Create new tracks for unmatched detections
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    new_track = Track(detection, self.next_id)
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1
                    
        else:
            # No existing tracks, create new ones for all detections
            for detection in detections:
                new_track = Track(detection, self.next_id)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            del self.tracks[track_id]
            
        return list(self.tracks.values())