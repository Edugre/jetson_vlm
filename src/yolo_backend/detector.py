import os
import cv2
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, source="/dev/video0", engine="yolov8n.engine",
                 imgsz=416, device="cuda:0", half=True, confidence_threshold=0.5):
        self.source = source
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.confidence_threshold = confidence_threshold
        
        # Prefer TensorRT engine if present, else .pt
        self.model = YOLO(engine) if os.path.exists(engine) else YOLO("yolov8n.pt")
        # names can be dict or list depending on backend
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))
        self.cap = None
        
        # Person detection state management
        self.person_in_frame = False
        self.last_person_count = 0
        self.detection_cooldown = 1.0  # Minimum seconds between notifications
        self.last_detection_time = 0
        self.detection_callbacks = []
        self.current_frame = None
        self.frame_lock = threading.Lock()

    def _name_of(self, cls_idx: int):
        if isinstance(self.names, dict):
            return self.names.get(cls_idx, str(cls_idx))
        if isinstance(self.names, (list, tuple)) and 0 <= cls_idx < len(self.names):
            return self.names[cls_idx]
        return str(cls_idx)
    
    def add_detection_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function for person detection events"""
        self.detection_callbacks.append(callback)
    
    def _trigger_callbacks(self, event_type: str, person_count: int, persons: List[Dict[str, Any]] = None):
        """Trigger all registered callbacks"""
        event_data = {
            "event": event_type,
            "person_count": person_count,
            "timestamp": time.time(),
            "persons": persons or []
        }
        
        for callback in self.detection_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in detection callback: {e}")
    
    def _check_person_events(self, current_person_count: int, persons: List[Dict[str, Any]]):
        """Check for person entry/exit events and trigger callbacks"""
        current_time = time.time()
        
        # Cooldown to prevent spam notifications
        if current_time - self.last_detection_time < self.detection_cooldown:
            return
            
        # Person entered frame
        if current_person_count > 0 and not self.person_in_frame:
            self.person_in_frame = True
            self.last_detection_time = current_time
            self._trigger_callbacks("person_entered", current_person_count, persons)
            logger.info(f"Person entered frame - Count: {current_person_count}")
            
        # Person left frame
        elif current_person_count == 0 and self.person_in_frame:
            self.person_in_frame = False
            self.last_detection_time = current_time
            self._trigger_callbacks("person_left", 0)
            logger.info("All persons left frame")
            
        # Person count changed
        elif current_person_count != self.last_person_count and current_person_count > 0:
            self.last_detection_time = current_time
            self._trigger_callbacks("person_count_changed", current_person_count, persons)
            logger.info(f"Person count changed: {self.last_person_count} -> {current_person_count}")
        
        self.last_person_count = current_person_count
    
    def get_detection_status(self) -> Dict[str, Any]:
        """Get current person detection status"""
        return {
            "person_in_frame": self.person_in_frame,
            "last_person_count": self.last_person_count,
            "confidence_threshold": self.confidence_threshold,
            "detection_cooldown": self.detection_cooldown
        }

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
                        confidence = float(conf.item())
                        if self._name_of(c) == "person" and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = [float(v.item()) for v in xyxy]
                            person_data = {
                                "cls": "person", 
                                "conf": confidence, 
                                "bbox": [x1, y1, x2, y2],
                                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                                "area": (x2 - x1) * (y2 - y1)
                            }
                            persons.append(person_data)
                
                # Store current frame for callbacks
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Check for person entry/exit events
                self._check_person_events(len(persons), persons)
                
                h, w = result.orig_shape  # (H, W)
                yield {"persons": persons, "frame": {"w": w, "h": h}}
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None