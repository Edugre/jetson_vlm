import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="EVE – Jetson YOLO", layout="wide")

st.title("🎥 EVE Security System – YOLOv8 Live Detection")

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # small model
    return model

model = load_model()
source = st.selectbox("Select camera source:", ["/dev/video0", "/dev/video1"], index=1)
run = st.checkbox("Run detection", value=False)

if run:
    stframe = st.empty()
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Camera feed ended.")
            break
        result = model.predict(source=frame, device="cuda:0", imgsz=416, half=True, verbose=False)
        annotated = result[0].plot()
        stframe.image(annotated, channels="BGR", use_column_width=True)
    cap.release()