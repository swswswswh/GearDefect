import cv2
import torch
from ultralytics import YOLO


class GearDetector:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.classes = ['pitting', 'indentation', 'scuffing', 'spalling']

    def detect(self, img_path):
        results = self.model(img_path)
        return self._parse_results(results)

    def _parse_results(self, results):
        defects = []
        for box in results.boxes:
            defects.append({
                'type': self.classes[int(box.cls)],
                'confidence': float(box.conf),
                'position': box.xyxy.tolist()
            })
        return defects
