#!/usr/bin/env python3
"""
YOLO11 CSI Camera Integration for Jetson Nano
Uses GStreamer pipeline to access CSI camera and run YOLO11 inference
"""

from ultralytics import YOLO
import cv2
import argparse


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

    Args:
        sensor_id: Camera sensor ID (0 for primary CSI camera)
        capture_width: Native capture width
        capture_height: Native capture height
        display_width: Output display width
        display_height: Output display height
        framerate: Camera framerate
        flip_method: Flip method (0=none, 1=ccw, 2=180, 3=cw, 4=horizontal, 5=upper-right, 6=vertical, 7=upper-left)

    Returns:
        GStreamer pipeline string
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 on Jetson CSI Camera")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)")
    parser.add_argument("--sensor-id", type=int, default=0,
                        help="CSI camera sensor ID (default: 0)")
    parser.add_argument("--capture-width", type=int, default=1280,
                        help="Camera capture width (default: 1280)")
    parser.add_argument("--capture-height", type=int, default=720,
                        help="Camera capture height (default: 720)")
    parser.add_argument("--display-width", type=int, default=640,
                        help="Display width (default: 640)")
    parser.add_argument("--display-height", type=int, default=480,
                        help="Display height (default: 480)")
    parser.add_argument("--framerate", type=int, default=30,
                        help="Camera framerate (default: 30)")
    parser.add_argument("--flip-method", type=int, default=0,
                        help="Flip method 0-7 (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--show", action="store_true",
                        help="Display the output video")
    parser.add_argument("--save", action="store_true",
                        help="Save the output video")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output")

    args = parser.parse_args()

    print(f"Loading YOLO11 model: {args.model}")
    model = YOLO(args.model)

    print("Opening CSI camera with GStreamer pipeline...")
    pipeline = gstreamer_pipeline(
        sensor_id=args.sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.display_width,
        display_height=args.display_height,
        framerate=args.framerate,
        flip_method=args.flip_method
    )

    if args.verbose:
        print(f"GStreamer pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Cannot open CSI camera")
        print("Make sure the camera is properly connected and recognized by the system")
        print("You can test with: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! xvimagesink")
        return

    print("CSI camera opened successfully!")
    print(f"Resolution: {args.display_width}x{args.display_height}")
    print("Press 'q' to quit")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame")
                break

            # Run YOLO11 inference
            results = model(frame, conf=args.conf, iou=args.iou, verbose=False)

            # Get annotated frame
            annotated_frame = results[0].plot()

            # Print detection info
            if args.verbose:
                boxes = results[0].boxes
                print(f"Frame {frame_count}: Detected {len(boxes)} objects")

            # Display frame
            if args.show:
                cv2.imshow('YOLO11 CSI Camera', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print(f"Total frames processed: {frame_count}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
