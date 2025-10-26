import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import streamlit as st
from ultralytics import YOLO


def choose_default_model():
    candidates = ["yolov8n.engine", "yolov8n.pt", "yolov8n.onnx", "yolo11n.pt"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolov8n.pt"


st.set_page_config(page_title="Security Anomaly Cam", layout="wide")
st.title("Security Anomaly Detection System")


model_path = choose_default_model()

# Auto-detect device; do not expose in UI (simple app)
cuda_available = torch.cuda.is_available()
selected_device = "cuda:0" if cuda_available else "cpu"

# Fixed parameters for simple UI
conf_thresh = 0.45
source = 0
tracker_config = "bytetrack.yaml"


frame_window = st.image([])
log_box = st.empty()

# Tracking and logs
last_seen = defaultdict(lambda: time.time())
object_history = defaultdict(list)
anomaly_log = deque(maxlen=512)
prev_frame = None
recent_events = set()

os.makedirs("logs", exist_ok=True)
log_file = "logs/anomalies.txt"

MOTION_THRESHOLD = 1e6
VANISH_TIME = 3.0
EVENT_COOLDOWN = 2.0


def log_event(text):
    now = time.time()
    event_key = (text, int(now // EVENT_COOLDOWN))
    if event_key in recent_events:
        return
    recent_events.add(event_key)
    entry = f"[{time.strftime('%H:%M:%S')}] {text}"
    anomaly_log.append(entry)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def _draw_boxes_and_extract_ids(model, result, frame, only_persons=True, conf_threshold=0.45):
    """Draw boxes on the frame and return tracking ids.

    When only_persons is True, only boxes labeled 'person' (case-insensitive)
    and with confidence >= conf_threshold will be drawn/returned.
    """
    ids = []
    names = getattr(model.model, "names", getattr(model, "names", {}))
    if result.boxes is None:
        return ids

    xyxy = getattr(result.boxes, 'xyxy', None)
    cls_iter = getattr(result.boxes, 'cls', None)
    conf_iter = getattr(result.boxes, 'conf', None)
    id_iter = getattr(result.boxes, 'id', None)

    if xyxy is None or cls_iter is None or conf_iter is None:
        return ids

    drawn_indices = []
    for idx, (box, cls, conf) in enumerate(zip(xyxy, cls_iter, conf_iter)):
        try:
            confidence = float(conf.item()) if hasattr(conf, 'item') else float(conf)
        except Exception:
            continue
        if confidence < conf_threshold:
            continue
        try:
            c = int(cls.item()) if hasattr(cls, 'item') else int(cls)
        except Exception:
            continue
        label_name = names.get(c, str(c)) if isinstance(names, dict) else (names[c] if 0 <= c < len(names) else str(c))
        if only_persons and str(label_name).lower() != 'person':
            continue
        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            continue
        label = f"{label_name} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        drawn_indices.append(idx)

    # attempt to read ids if available (only return ids for drawn persons)
    if id_iter is not None:
        try:
            raw_ids = result.boxes.id.int().tolist()
        except Exception:
            raw_ids = None

        for di in drawn_indices:
            if raw_ids is not None and di < len(raw_ids):
                try:
                    ids.append(int(raw_ids[di]))
                except Exception:
                    pass
    return ids


def run_app():
    st.write("### Security Anomaly Detection")

    # Load model once
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        return

    # model.track with tracker (ByteTrack) — starts immediately
    try:
        prev_frame_local = None
        for result in model.track(source=source, tracker=tracker_config, stream=True, conf=conf_thresh, device=selected_device):
            frame = result.orig_img
            if frame is None or getattr(frame, 'size', 0) == 0:
                continue

            now = time.time()
            anomalies = []

            # scene motion
            if prev_frame_local is not None:
                try:
                    gray1 = cv2.cvtColor(prev_frame_local, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray1, gray2)
                    motion = np.sum(diff > 25)
                    if motion > MOTION_THRESHOLD:
                        anomalies.append("Large sudden scene motion detected (possible intrusion).")
                except Exception:
                    pass
            prev_frame_local = frame.copy()

            # draw person boxes and get tracker ids
            ids = _draw_boxes_and_extract_ids(model, result, frame, only_persons=True, conf_threshold=conf_thresh)

            for oid in ids:
                if oid not in last_seen:
                    anomalies.append(f"New object detected (ID {oid}).")
                last_seen[oid] = now
                object_history[oid].append(now)

            vanished = [oid for oid, t in list(last_seen.items()) if now - t > VANISH_TIME]
            for oid in vanished:
                anomalies.append(f"Object {oid} disappeared from view.")
                try:
                    del last_seen[oid]
                except KeyError:
                    pass

            for text in anomalies:
                log_event(text)

            # show frame and last log lines
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                frame_rgb = frame
            frame_window.image(frame_rgb, channels="RGB")
            last_entries = "\n".join(list(anomaly_log)[-12:]) if anomaly_log else "No events yet."
            log_box.markdown(f"### Live Anomaly Log\n```\n{last_entries}\n```")

            # tiny sleep so Streamlit can update UI
            time.sleep(0.03)
    except Exception as e:
        st.error(f"Streaming stopped due to error: {e}")


if __name__ == "__main__":
    run_app()
