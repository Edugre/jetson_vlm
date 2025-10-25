#!/usr/bin/env python3
"""
Streamlit App for YOLO11 CSI Camera Live Video
Real-time object detection with web interface
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    """
    Create GStreamer pipeline for CSI camera on Jetson Nano
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


@st.cache_resource
def load_model(model_name):
    """Load YOLO model (cached)"""
    return YOLO(model_name)


@st.cache_resource
def init_camera(sensor_id, capture_width, capture_height, display_width, display_height, framerate, flip_method):
    """Initialize CSI camera (cached)"""
    pipeline = gstreamer_pipeline(
        sensor_id=sensor_id,
        capture_width=capture_width,
        capture_height=capture_height,
        display_width=display_width,
        display_height=display_height,
        framerate=framerate,
        flip_method=flip_method
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def main():
    st.set_page_config(
        page_title="YOLO11 Live Detection",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 YOLO11 CSI Camera Live Detection")
    st.markdown("Real-time object detection on Jetson Nano")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    # Model selection
    model_options = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    selected_model = st.sidebar.selectbox("YOLO Model", model_options, index=0)

    # Camera settings
    st.sidebar.subheader("Camera Settings")
    sensor_id = st.sidebar.number_input("Sensor ID", 0, 1, 0)
    flip_method = st.sidebar.selectbox("Flip Method", [0, 1, 2, 3, 4, 5, 6, 7], index=0)

    # Resolution settings
    st.sidebar.subheader("Resolution")
    display_width = st.sidebar.slider("Display Width", 320, 1280, 320, 160)
    display_height = st.sidebar.slider("Display Height", 240, 720, 240, 120)

    # Performance settings
    st.sidebar.subheader("Performance")
    skip_frames = st.sidebar.slider("Skip Frames (process every Nth frame)", 1, 10, 3, 1)
    update_interval = st.sidebar.slider("UI Update Interval (seconds)", 0.1, 2.0, 0.5, 0.1)

    # Detection settings
    st.sidebar.subheader("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

    # Control buttons
    start_button = st.sidebar.button("🎬 Start Detection")
    stop_button = st.sidebar.button("⏹️ Stop Detection")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📊 Detection Stats")
        fps_text = st.empty()
        detection_count = st.empty()
        detected_objects = st.empty()

    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False

    if start_button:
        st.session_state.running = True

    if stop_button:
        st.session_state.running = False

    # Run detection loop
    if st.session_state.running:
        try:
            # Load model
            with st.spinner(f"Loading {selected_model}..."):
                model = load_model(selected_model)

            # Initialize camera
            with st.spinner("Initializing CSI camera..."):
                cap = init_camera(
                    sensor_id=sensor_id,
                    capture_width=1280,
                    capture_height=720,
                    display_width=display_width,
                    display_height=display_height,
                    framerate=30,
                    flip_method=flip_method
                )

            if not cap.isOpened():
                st.error("❌ Cannot open CSI camera. Please check connections.")
                st.session_state.running = False
                return

            st.success("✅ Camera initialized successfully!")

            frame_count = 0
            fps_start_time = time.time()
            fps = 0
            last_update_time = time.time()

            # Variables to hold last results
            last_annotated_frame = None
            last_num_detections = 0
            last_detected_classes = {}

            while st.session_state.running:
                ret, frame = cap.read()

                if not ret:
                    st.error("❌ Failed to capture frame")
                    break

                frame_count += 1

                # Skip frames for better performance
                if frame_count % skip_frames != 0:
                    continue

                # Run YOLO inference only on selected frames
                results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

                # Get annotated frame
                annotated_frame = results[0].plot()

                # Get detection info
                boxes = results[0].boxes
                num_detections = len(boxes)

                # Calculate FPS
                fps_end_time = time.time()
                fps = skip_frames / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time

                # Collect detected objects
                detected_classes = {}
                if num_detections > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])

                        if cls_name in detected_classes:
                            detected_classes[cls_name].append(conf)
                        else:
                            detected_classes[cls_name] = [conf]

                # Store results
                last_annotated_frame = annotated_frame
                last_num_detections = num_detections
                last_detected_classes = detected_classes

                # Update UI only at specified intervals to reduce overhead
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # Convert BGR to RGB for display
                    annotated_frame_rgb = cv2.cvtColor(last_annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display frame
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

                    # Update stats
                    fps_text.metric("FPS", f"{fps:.1f}")
                    detection_count.metric("Objects Detected", last_num_detections)

                    # Show detected objects
                    if last_num_detections > 0:
                        objects_text = ""
                        for cls_name, confs in last_detected_classes.items():
                            avg_conf = np.mean(confs)
                            objects_text += f"**{cls_name}**: {len(confs)} (conf: {avg_conf:.2f})\n\n"
                        detected_objects.markdown(objects_text)
                    else:
                        detected_objects.markdown("*No objects detected*")

                    last_update_time = current_time

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.running = False

        finally:
            if 'cap' in locals():
                cap.release()

    else:
        # Show instructions when not running
        with col1:
            st.info("👈 Click **Start Detection** in the sidebar to begin live video streaming")

        with col2:
            st.markdown("""
            ### How to Use:
            1. Select YOLO model (nano is fastest)
            2. **Lower resolution to 320x240** for best performance
            3. **Increase skip frames** to process fewer frames
            4. Adjust UI update interval
            5. Click **Start Detection**

            ### Performance Tips for Jetson Nano:
            - Use **yolo11n.pt** (nano model)
            - Set resolution to **320x240**
            - Skip frames: **3-5** (process every 3rd-5th frame)
            - UI update: **0.5-1.0 seconds**
            - Lower confidence = more detections but slower
            """)


if __name__ == "__main__":
    main()
