import os
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

    def _name_of(self, cls_idx: int):
        if isinstance(self.names, dict):
            return self.names.get(cls_idx, str(cls_idx))
        if isinstance(self.names, (list, tuple)) and 0 <= cls_idx < len(self.names):
            return self.names[cls_idx]
        return str(cls_idx)

    def stream(self):
        # Always stream to keep memory stable on Jetson
        for result in self.model.predict(
            source=self.source, device=self.device, stream=True,
            imgsz=self.imgsz, half=self.half, verbose=False, warmup=False
        ):
            persons = []
            if result.boxes is not None and len(result.boxes):
                for xyxy, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    c = int(cls.item())
                    if self._name_of(c) == "person":
                        x1, y1, x2, y2 = [float(v.item()) for v in xyxy]
                        persons.append({"cls": "person", "conf": float(conf.item()), "bbox": [x1, y1, x2, y2]})
            h, w = result.orig_shape  # (H, W)
            yield {"persons": persons, "frame": {"w": w, "h": h}}