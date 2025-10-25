#!/usr/bin/env python3
"""
YOLOv8 USB Camera with Streamlit Web Interface (TensorRT Optimized)
Real-time object detection displayed in web browser with TensorRT acceleration
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
import os
from pathlib import Path
import torch


def check_cuda_available():
    """Check if CUDA is available"""
    return torch.cuda.is_available()


@st.cache_resource
def load_model(model_path, use_tensorrt=False):
    """Load and cache the YOLO model, optionally with TensorRT optimization"""
    if use_tensorrt:
        # Check if CUDA is available
        if not check_cuda_available():
            st.sidebar.error("❌ CUDA not available. TensorRT requires CUDA-enabled PyTorch.")
            st.sidebar.info("""
            **To enable CUDA on Jetson:**
            1. Install PyTorch with CUDA support from NVIDIA
            2. For Jetson, use: https://forums.developer.nvidia.com/t/pytorch-for-jetson/
            3. Or install using NVIDIA's wheel files
            """)
            st.sidebar.warning("⚠️ Falling back to CPU inference")
            return YOLO(model_path)

        # Check if TensorRT engine exists
        engine_path = model_path.replace('.pt', '.engine')

        if os.path.exists(engine_path):
            st.sidebar.info(f"✅ Loading existing TensorRT engine: {engine_path}")
            return YOLO(engine_path)
        else:
            st.sidebar.warning(f"TensorRT engine not found. Exporting {model_path} to TensorRT...")
            st.sidebar.info("⏳ This may take a few minutes on first run...")
            model = YOLO(model_path)

            # Export to TensorRT
            try:
                # Use device=0 since CUDA is available
                model.export(format='engine', device=0, half=True)
                st.sidebar.success(f"✅ TensorRT engine created: {engine_path}")
                return YOLO(engine_path)
            except Exception as e:
                st.sidebar.error(f"❌ Failed to export to TensorRT: {str(e)}")
                st.sidebar.info("⚠️ Falling back to PyTorch model")
                return model
    else:
        return YOLO(model_path)


def main():
    st.set_page_config(
        page_title="YOLOv8 Camera Detection",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 YOLOv8 Real-Time Object Detection")
    st.markdown("Live camera feed with AI-powered object detection")

    # Sidebar controls
    st.sidebar.header("Settings")

    # Display CUDA status
    cuda_available = check_cuda_available()
    if cuda_available:
        st.sidebar.success(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("⚠️ CUDA Not Available - CPU mode only")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Smaller models are faster, larger models are more accurate"
    )

    # TensorRT optimization
    use_tensorrt = st.sidebar.checkbox(
        "Use TensorRT",
        value=True,
        help="Enable TensorRT acceleration for faster inference on Jetson (recommended)"
    )

    # Camera settings
    device_id = st.sidebar.number_input(
        "Camera Device ID",
        min_value=0,
        max_value=10,
        value=1,
        help="Usually 1 for /dev/video1"
    )

    width = st.sidebar.number_input("Width", min_value=320, max_value=1920, value=640, step=160)
    height = st.sidebar.number_input("Height", min_value=240, max_value=1080, value=480, step=120)

    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )

    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression"
    )

    # FPS display toggle
    show_fps = st.sidebar.checkbox("Show FPS", value=True)

    # Start/Stop button
    run_camera = st.sidebar.checkbox("Start Camera", value=False)

    # Load model
    loading_msg = f"Loading {model_choice}" + (" with TensorRT..." if use_tensorrt else "...")
    with st.spinner(loading_msg):
        model = load_model(model_choice, use_tensorrt)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Camera Feed")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("Detection Statistics")
        stats_placeholder = st.empty()
        fps_placeholder = st.empty()

    if run_camera:
        # Open camera
        cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            st.error(f"❌ Cannot open camera at /dev/video{device_id}")
            st.info("Try running: `v4l2-ctl --list-devices` to see available cameras")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.sidebar.success(f"✅ Camera opened: {actual_width}x{actual_height}")

        frame_count = 0
        start_time = time.time()
        fps = 0

        try:
            while run_camera:
                ret, frame = cap.read()

                if not ret:
                    st.error("❌ Failed to capture frame")
                    break

                # Run YOLO inference
                results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

                # Get annotated frame
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time

                # Add FPS to frame if enabled
                if show_fps:
                    cv2.putText(
                        annotated_frame_rgb,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                # Display frame
                frame_placeholder.image(annotated_frame_rgb, channels="RGB", width="stretch")

                # Display statistics
                boxes = results[0].boxes
                num_detections = len(boxes)

                # Count detections by class
                if num_detections > 0:
                    class_counts = {}
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # Display stats
                    stats_text = f"**Total Objects:** {num_detections}\n\n"
                    stats_text += "**Detected Classes:**\n"
                    for class_name, count in sorted(class_counts.items()):
                        stats_text += f"- {class_name}: {count}\n"

                    stats_placeholder.markdown(stats_text)
                else:
                    stats_placeholder.info("No objects detected")

                # Display FPS
                if show_fps:
                    fps_placeholder.metric("Frame Rate", f"{fps:.1f} FPS")

                # Small delay to allow Streamlit to update
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Error: {str(e)}")

        finally:
            cap.release()
            st.sidebar.info(f"Total frames processed: {frame_count}")

    else:
        st.info("👈 Click 'Start Camera' in the sidebar to begin detection")
        st.markdown("""
        ### How to use:
        1. Select your preferred YOLO model (smaller = faster, larger = more accurate)
        2. Enable TensorRT for maximum performance (recommended for Jetson)
        3. Adjust the camera device ID if needed (usually 1 for /dev/video1)
        4. Set your desired resolution
        5. Tune detection parameters (confidence and IoU thresholds)
        6. Click "Start Camera" to begin!

        ### TensorRT Acceleration:
        - **First run**: Model will be exported to TensorRT engine (takes a few minutes)
        - **Subsequent runs**: Uses cached TensorRT engine for instant loading
        - **Performance**: 2-5x faster inference compared to PyTorch
        - **FP16 precision**: Uses half-precision for better speed on Jetson

        ### Tips:
        - Lower confidence threshold = more detections (but more false positives)
        - Higher IoU threshold = less filtering of overlapping boxes
        - Smaller models (yolov8n) run faster on Jetson devices
        - Enable TensorRT for best performance on NVIDIA Jetson hardware
        """)


if __name__ == "__main__":
    main()
