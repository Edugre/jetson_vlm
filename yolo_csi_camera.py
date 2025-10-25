#!/usr/bin/env python3
"""
YOLO11 USB Camera Integration for Jetson Nano
Uses USB camera to run YOLO11 inference
"""

from ultralytics import YOLO
import cv2
import argparse




def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 on Jetson USB Camera")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)")
    parser.add_argument("--device-id", type=int, default=1,
                        help="USB camera device ID (default: 1, usually /dev/video1)")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera height (default: 480)")
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

    print(f"Opening USB camera (device {args.device_id})...")
    cap = cv2.VideoCapture(args.device_id)

    if not cap.isOpened():
        print(f"Error: Cannot open USB camera at /dev/video{args.device_id}")
        print("Make sure the camera is properly connected and recognized by the system")
        print("You can list available cameras with: v4l2-ctl --list-devices")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get actual resolution (camera may not support requested resolution)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("USB camera opened successfully!")
    print(f"Resolution: {actual_width}x{actual_height}")
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
                cv2.imshow('YOLO11 USB Camera', annotated_frame)

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
