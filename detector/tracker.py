"""Centroid tracker + kinematic feature extraction.

Per frame we reduce the scene to a fixed-size feature vector (NUM_FEATURES).
A window of SEQ_LEN such vectors is the input of the BiGRU classifier.
Speeds / decelerations are normalised by the frame diagonal (value = % of the
diagonal moved per frame) so the model does not depend on video resolution.
"""
import math
from collections import deque
import numpy as np

SEQ_LEN = 16
FEATURE_NAMES = ["max_speed", "mean_speed", "max_decel", "max_iou", "min_dist", "n_vehicles"]
NUM_FEATURES = len(FEATURE_NAMES)


def iou(a, b):
    """Intersection over Union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.history = deque([self.centroid], maxlen=30)
        self.speed = 0.0      # pixels / frame (smoothed)
        self.decel = 0.0      # pixels / frame^2 (only drops in speed)
        self.missing = 0
        self.age = 1

    def update(self, box):
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        dist = math.hypot(c[0] - self.centroid[0], c[1] - self.centroid[1])
        new_speed = dist if self.age == 1 else 0.4 * self.speed + 0.6 * dist
        self.decel = max(0.0, self.speed - new_speed) if self.age > 1 else 0.0
        self.speed = new_speed
        self.box, self.centroid = box, c
        self.history.append(c)
        self.missing = 0
        self.age += 1


class CentroidTracker:
    def __init__(self, max_distance=90, max_missing=8):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes):
        """boxes: list of (x1, y1, x2, y2). Returns list of currently visible tracks."""
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        pairs = []
        for tid, t in self.tracks.items():
            for j, c in enumerate(cents):
                d = math.hypot(c[0] - t.centroid[0], c[1] - t.centroid[1])
                if d <= self.max_distance:
                    pairs.append((d, tid, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for d, tid, j in pairs:                     # greedy nearest-first matching
            if tid in used_t or j in used_d:
                continue
            self.tracks[tid].update(boxes[j])
            used_t.add(tid); used_d.add(j)
        for tid, t in list(self.tracks.items()):    # unmatched tracks
            if tid not in used_t:
                t.missing += 1
                if t.missing > self.max_missing:
                    del self.tracks[tid]
        for j, b in enumerate(boxes):               # new tracks
            if j not in used_d:
                self.tracks[self.next_id] = Track(self.next_id, b)
                self.next_id += 1
        return [t for t in self.tracks.values() if t.missing == 0]


def frame_features(tracks, frame_shape):
    """Return a NUM_FEATURES vector describing the current frame."""
    h, w = frame_shape[:2]
    diag = math.hypot(w, h)
    k = 100.0 / diag
    if not tracks:
        return np.array([0, 0, 0, 0, 1.0, 0], dtype=np.float32)
    speeds = [t.speed * k for t in tracks]
    decels = [t.decel * k for t in tracks]
    max_iou, min_dist = 0.0, 1.0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            max_iou = max(max_iou, iou(tracks[i].box, tracks[j].box))
            d = math.hypot(tracks[i].centroid[0] - tracks[j].centroid[0],
                           tracks[i].centroid[1] - tracks[j].centroid[1]) / diag
            min_dist = min(min_dist, d)
    return np.array([max(speeds), float(np.mean(speeds)), max(decels), max_iou,
                     min_dist, min(len(tracks) / 10.0, 1.0)], dtype=np.float32)
