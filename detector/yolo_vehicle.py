"""Vehicle detectors.

YoloVehicleDetector : pretrained YOLOv8 (COCO) - use for real traffic video.
ColorBlobDetector   : simple colour-segmentation detector - ONLY for the
                      synthetic demo videos made by generate_sample_video.py
                      (YOLO cannot recognise the drawn rectangles as vehicles).
Both return a list of (x1, y1, x2, y2, confidence, label).
"""
import cv2
import numpy as np

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloVehicleDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.35):
        from ultralytics import YOLO          # imported lazily
        self.model = YOLO(weights)            # downloads weights on first use
        self.conf = conf

    def detect(self, frame):
        res = self.model(frame, verbose=False, conf=self.conf,
                         classes=list(VEHICLE_CLASSES))[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(b.conf[0]), VEHICLE_CLASSES[int(b.cls[0])]))
        return out


class ColorBlobDetector:
    # OpenCV hue ranges (0-179) for the coloured cars drawn in the sample videos
    HUES = [((0, 8), (172, 179)), ((20, 35),), ((40, 85),), ((100, 130),)]

    def __init__(self, min_area=400):
        self.min_area = min_area
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_ok = (hsv[..., 1] > 110) & (hsv[..., 2] > 70)
        out = []
        for ranges in self.HUES:                       # one mask per colour so that
            mask = np.zeros(frame.shape[:2], np.uint8)  # touching cars stay separate
            for lo, hi in ranges:
                mask |= ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)).astype(np.uint8)
            mask = (mask & sat_ok).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= self.min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    out.append((x, y, x + w, y + h, 1.0, "car"))
        return out


def make_detector(kind="yolo"):
    return ColorBlobDetector() if kind == "blob" else YoloVehicleDetector()
