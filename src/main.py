# main.py
from yolo_backend.detector import YOLODetector
from adk_backend.servo_controller import ServoController
from utils.logger import log

# Tuning knobs (safe defaults for Jetson)
IMG_SZ = 416
PAN_STEP_SMALL = 3      # degrees per nudge
TILT_STEP_SMALL = 3
CENTER_X_MARGIN = 0.10  # 10% left/right deadband
CENTER_Y_MARGIN = 0.10  # 10% up/down deadband

def nudge_to_center(servo, frame_w, frame_h, bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Horizontal
    left_border  = frame_w * (0.5 - CENTER_X_MARGIN)
    right_border = frame_w * (0.5 + CENTER_X_MARGIN)
    if cx < left_border:
        servo.move_left(PAN_STEP_SMALL)
        side = "left"
    elif cx > right_border:
        servo.move_right(PAN_STEP_SMALL)
        side = "right"
    else:
        side = "center-x"

    # Vertical (y=0 is top)
    top_border    = frame_h * (0.5 - CENTER_Y_MARGIN)
    bottom_border = frame_h * (0.5 + CENTER_Y_MARGIN)
    if cy < top_border:
        servo.move_up(TILT_STEP_SMALL)
        vert = "up"
    elif cy > bottom_border:
        servo.move_down(TILT_STEP_SMALL)
        vert = "down"
    else:
        vert = "center-y"

    return side, vert

def main():
    det = YOLODetector(source="/dev/video0", imgsz=IMG_SZ, device="cuda:0", half=True)
    servo = ServoController()
    had_person = False

    for pack in det.stream():
        persons = pack["persons"]
        frame = pack["frame"]

        if persons:
            # Track the most confident person
            person = max(persons, key=lambda d: d["conf"])
            side, vert = nudge_to_center(servo, frame["w"], frame["h"], person["bbox"])

            # Fire "anomaly" only on 0 -> 1 transition (new person appeared)
            if not had_person:
                log(f"Anomaly: person entered (first seen). Hint side={side}, vert={vert}")
                had_person = True
        else:
            if had_person:
                log("Info: person left frame.")
            had_person = False

if __name__ == "__main__":
    main()