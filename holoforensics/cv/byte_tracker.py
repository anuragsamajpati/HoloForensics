import numpy as np
from collections import OrderedDict

class STrack:
    def __init__(self, tlwh, score, cls_id):
        self.tlwh = np.asarray(tlwh, dtype=float)  # Fixed: removed float
        self.score = score
        self.cls_id = cls_id
        self.track_id = 0
        self.is_activated = False
        self.state = "new"

    def activate(self, track_id):
        self.track_id = track_id
        self.is_activated = True
        self.state = "tracked"

class SimpleTracker:
    def __init__(self):
        self.tracks = OrderedDict()
        self.next_id = 1

    def update(self, detections):
        tracks = []
        for det in detections:
            track = STrack(det["bbox"], det["confidence"], det.get("class_id", 0))
            track.activate(self.next_id)
            self.next_id += 1
            tracks.append(track)
        return tracks
