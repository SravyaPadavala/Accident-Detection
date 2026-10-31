"""Detection -> tracking -> features -> classifier -> alert / evidence / DB."""
import os
import time
from collections import deque
import cv2
import numpy as np
import database
from .tracker import CentroidTracker, frame_features, SEQ_LEN
from .yolo_vehicle import make_detector
from .accident_model import load_classifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR = os.path.join(BASE, "static", "evidence")
THRESHOLD = 0.65          # decision threshold on P(accident)
COOLDOWN = 4.0            # seconds between two stored alerts


class AccidentPipeline:
    def __init__(self, detector="yolo", threshold=THRESHOLD, cooldown=COOLDOWN,
                 classifier="auto", source="video", evidence_dir=EVIDENCE_DIR):
        self.detector = make_detector(detector) if isinstance(detector, str) else detector
        self.classifier = load_classifier() if classifier == "auto" else classifier
        self.threshold, self.cooldown, self.source = threshold, cooldown, source
        self.evidence_dir = evidence_dir
        self.tracker = CentroidTracker()
        self.buffer = deque(maxlen=SEQ_LEN)
        self.last_alert = None
        self.alert_until = -1.0
        self.frames = 0
        self.max_prob = 0.0
        self.incidents = []
        self.vehicles = 0
        self.prob = 0.0

    @property
    def classifier_available(self):
        return self.classifier is not None

    def process_frame(self, frame, now):
        """Returns annotated frame. `now` = time in seconds (video time or wall time)."""
        boxes = self.detector.detect(frame)
        tracks = self.tracker.update([b[:4] for b in boxes])
        self.buffer.append(frame_features(tracks, frame.shape))
        self.frames += 1
        self.vehicles = len(tracks)
        prob = None
        if self.classifier and len(self.buffer) == SEQ_LEN:
            prob = self.classifier.predict(np.array(self.buffer))
            self.prob = prob
            self.max_prob = max(self.max_prob, prob)
        alert = False
        if prob is not None and prob >= self.threshold and \
                (self.last_alert is None or now - self.last_alert >= self.cooldown):
            self.last_alert = now
            self.alert_until = now + 2.0
            alert = True
        out = self._annotate(frame, tracks, prob, now)
        if alert:
            self.incidents.append(self._save_incident(out, prob))
        return out

    def _save_incident(self, frame, prob):
        os.makedirs(self.evidence_dir, exist_ok=True)
        name = f"incident_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(self.evidence_dir, name), frame)
        iid = database.add_incident(prob, f"evidence/{name}", self.source, self.vehicles)
        return {"id": iid, "confidence": round(prob, 3), "evidence": f"evidence/{name}"}

    def _annotate(self, frame, tracks, prob, now):
        img = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID {t.id}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if not self.classifier:
            txt = "Classifier unavailable (train model first)"
        elif prob is None:
            txt = "Collecting frames..."
        else:
            txt = f"P(accident) = {prob:.2f}"
        cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if now <= self.alert_until:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 6)
            cv2.putText(img, "POSSIBLE ACCIDENT", (img.shape[1] // 2 - 130, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        return img

    def process_video(self, path):
        """Analyse a whole video file (no display). Returns a summary dict."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.process_frame(frame, i / fps)
            i += 1
        cap.release()
        return {"frames": self.frames, "incidents": self.incidents,
                "max_probability": round(self.max_prob, 3),
                "classifier_available": self.classifier_available}
