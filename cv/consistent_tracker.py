import numpy as np
from collections import OrderedDict

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2
    x1_1, y1_1, x2_1, y2_1 = x1, y1, x1 + w1, y1 + h1
    x1_2, y1_2, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2
    
    # Calculate intersection
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0

class ConsistentTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=5):
        self.next_id = 1
        self.tracks = OrderedDict()  # track_id -> last_bbox
        self.disappeared = OrderedDict()  # track_id -> frames_disappeared
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
    
    def update(self, detections):
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)
            return []
        
        if len(self.tracks) == 0:
            # Initialize tracks for first detections
            for detection in detections:
                self.register(detection)
        else:
            # Match detections to existing tracks
            track_ids = list(self.tracks.keys())
            track_bboxes = list(self.tracks.values())
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(track_bboxes), len(detections)))
            for i, track_bbox in enumerate(track_bboxes):
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = calculate_iou(track_bbox, detection["bbox"])
            
            # Simple greedy matching
            used_detection_indices = set()
            used_track_indices = set()
            
            # Find best matches
            for i in range(len(track_ids)):
                if i in used_track_indices:
                    continue
                best_iou = 0
                best_j = -1
                for j in range(len(detections)):
                    if j in used_detection_indices:
                        continue
                    if iou_matrix[i, j] > best_iou and iou_matrix[i, j] > self.iou_threshold:
                        best_iou = iou_matrix[i, j]
                        best_j = j
                
                if best_j != -1:
                    # Update existing track
                    track_id = track_ids[i]
                    self.tracks[track_id] = detections[best_j]["bbox"]
                    self.disappeared[track_id] = 0
                    used_track_indices.add(i)
                    used_detection_indices.add(best_j)
            
            # Register new tracks for unmatched detections
            for j, detection in enumerate(detections):
                if j not in used_detection_indices:
                    self.register(detection)
            
            # Mark unmatched tracks as disappeared
            for i, track_id in enumerate(track_ids):
                if i not in used_track_indices:
                    self.disappeared[track_id] += 1
                    if self.disappeared[track_id] > self.max_disappeared:
                        self.deregister(track_id)
        
        # Return current tracks with their IDs
        active_tracks = []
        for track_id, bbox in self.tracks.items():
            active_tracks.append({"track_id": track_id, "bbox": bbox})
        return active_tracks
    
    def register(self, detection):
        self.tracks[self.next_id] = detection["bbox"]
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, track_id):
        del self.tracks[track_id]
        del self.disappeared[track_id]