import os
import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, source="/dev/video0", engine="yolov8n.engine",
                 imgsz=416, device="cuda:0", half=True):
        self.source = source
        self.imgsz = imgsz
        self.device = device
        self.half = half
        # Prefer TensorRT engine if present, else .pt
        self.model = YOLO(engine) if os.path.exists(engine) else YOLO("yolov8n.pt")
        # names can be dict or list depending on backend
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))
        self.cap = None

    def _name_of(self, cls_idx: int):
        if isinstance(self.names, dict):
            return self.names.get(cls_idx, str(cls_idx))
        if isinstance(self.names, (list, tuple)) and 0 <= cls_idx < len(self.names):
            return self.names[cls_idx]
        return str(cls_idx)

    def stream(self):
        # Manually capture frames to avoid cv2.waitKey() in headless environment
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Run prediction on single frame (stream=False to avoid LoadStreams)
                results = self.model.predict(
                    source=frame, device=self.device, stream=False,
                    imgsz=self.imgsz, half=self.half, verbose=False
                )

                # results is a list when stream=False, get first item
                result = results[0] if isinstance(results, list) else results

                persons = []
                if result.boxes is not None and len(result.boxes):
                    for xyxy, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        c = int(cls.item())
                        if self._name_of(c) == "person":
                            x1, y1, x2, y2 = [float(v.item()) for v in xyxy]
                            persons.append({"cls": "person", "conf": float(conf.item()), "bbox": [x1, y1, x2, y2]})
                h, w = result.orig_shape  # (H, W)
                yield {"persons": persons, "frame": {"w": w, "h": h}}
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None