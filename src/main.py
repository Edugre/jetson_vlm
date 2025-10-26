# main.py
import asyncio
import logging
from yolo_backend.detector import YOLODetector
from adk_backend.servo_controller import ServoController
from adk_backend.agent import root_agent
from notifications.person_detector import person_notifier
from utils.logger import log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tuning knobs (safe defaults for Jetson)
IMG_SZ = 416
PAN_STEP_SMALL = 3      # degrees per nudge
TILT_STEP_SMALL = 3
CENTER_X_MARGIN = 0.10  # 10% left/right deadband
CENTER_Y_MARGIN = 0.10  # 10% up/down deadband
CONFIDENCE_THRESHOLD = 0.6  # Higher confidence for fewer false positives

def nudge_to_center(servo, frame_w, frame_h, bbox, person_id="unknown"):
    """
    Nudge camera to center a person's bounding box.
    Returns detailed movement info for debugging.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    frame_center_x = frame_w / 2.0
    frame_center_y = frame_h / 2.0
    
    # Calculate offsets from center (in pixels and percentage)
    offset_x_px = cx - frame_center_x
    offset_y_px = cy - frame_center_y
    offset_x_pct = (offset_x_px / frame_w) * 100
    offset_y_pct = (offset_y_px / frame_h) * 100

    # Horizontal movement
    left_border  = frame_w * (0.5 - CENTER_X_MARGIN)
    right_border = frame_w * (0.5 + CENTER_X_MARGIN)
    
    pan_moved = False
    if cx < left_border:
        result = servo.move_left(PAN_STEP_SMALL)
        side = "LEFT"
        pan_moved = True
        logger.info(f"🎯 PAN LEFT {PAN_STEP_SMALL}° | Person {person_id} at x={cx:.0f} (needs {offset_x_px:.0f}px left) | New angle: {result['new_angle']}°")
    elif cx > right_border:
        result = servo.move_right(PAN_STEP_SMALL)
        side = "RIGHT"
        pan_moved = True
        logger.info(f"🎯 PAN RIGHT {PAN_STEP_SMALL}° | Person {person_id} at x={cx:.0f} (needs {offset_x_px:.0f}px right) | New angle: {result['new_angle']}°")
    else:
        side = "CENTERED-X"

    # Vertical movement (y=0 is top)
    top_border    = frame_h * (0.5 - CENTER_Y_MARGIN)
    bottom_border = frame_h * (0.5 + CENTER_Y_MARGIN)
    
    tilt_moved = False
    if cy < top_border:
        result = servo.move_up(TILT_STEP_SMALL)
        vert = "UP"
        tilt_moved = True
        logger.info(f"🎯 TILT UP {TILT_STEP_SMALL}° | Person {person_id} at y={cy:.0f} (needs {-offset_y_px:.0f}px up) | New angle: {result['new_angle']}°")
    elif cy > bottom_border:
        result = servo.move_down(TILT_STEP_SMALL)
        vert = "DOWN"
        tilt_moved = True
        logger.info(f"🎯 TILT DOWN {TILT_STEP_SMALL}° | Person {person_id} at y={cy:.0f} (needs {offset_y_px:.0f}px down) | New angle: {result['new_angle']}°")
    else:
        vert = "CENTERED-Y"
    
    if not pan_moved and not tilt_moved:
        logger.debug(f"✓ Person {person_id} already centered at ({cx:.0f}, {cy:.0f})")

    return side, vert, {
        "person_id": person_id,
        "center": (cx, cy),
        "offset_px": (offset_x_px, offset_y_px),
        "offset_pct": (offset_x_pct, offset_y_pct),
        "pan": side,
        "tilt": vert,
        "moved": pan_moved or tilt_moved
    }

def enhanced_person_callback(event_data):
    """Enhanced callback that provides person position data to the notifier"""
    try:
        persons = event_data.get("persons", [])
        if persons and event_data.get("event") == "person_entered":
            # Get the most confident person for tracking
            best_person = max(persons, key=lambda p: p.get("conf", 0))
            center = best_person.get("center", [320, 240])  # Default center
            
            # Add position data for the ADK agent
            event_data["primary_target_position"] = {
                "x": center[0],
                "y": center[1],
                "confidence": best_person.get("conf", 0)
            }
        
        # Forward to the person notifier (this will handle async internally)
        person_notifier.on_person_detected(event_data)
        
    except Exception as e:
        logger.error(f"Error in enhanced person callback: {e}")

def main():
    """Main function with integrated person detection and ADK agent"""
    logger.info("🤖 Starting EVE Security System...")
    
    # Initialize components
    det = YOLODetector(
        source="/dev/video0", 
        imgsz=IMG_SZ, 
        device="cuda:0", 
        half=True,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    servo = ServoController()
    
    # Connect person detection to the notifier
    det.add_detection_callback(enhanced_person_callback)
    
    logger.info("🎯 Person detection callbacks registered")
    logger.info("📹 Starting YOLO detection stream...")
    logger.info(f"⚙️  Settings: PAN_STEP={PAN_STEP_SMALL}°, TILT_STEP={TILT_STEP_SMALL}°, CENTER_MARGIN={CENTER_X_MARGIN*100}%")
    
    had_person = False
    newest_person = None  # Track the first/newest person detected
    frame_count = 0

    try:
        for pack in det.stream():
            persons = pack["persons"]
            frame = pack["frame"]
            frame_count += 1

            if persons:
                # If this is the first frame with people, pick the newest (first detected)
                if not had_person:
                    # Sort by area (largest first) to pick the most prominent person as "newest"
                    newest_person = max(persons, key=lambda d: d["area"])
                    person_id = f"P{frame_count}"
                    logger.info(f"🆕 NEW PERSON DETECTED! Person {person_id} | Area: {newest_person['area']:.0f}px² | Conf: {newest_person['conf']:.2f}")
                    logger.info(f"📊 Total people in frame: {len(persons)}")
                    had_person = True
                    
                    # Log frame info
                    logger.info(f"📐 Frame dimensions: {frame['w']}x{frame['h']}")
                    logger.info(f"📍 Person center: ({newest_person['center'][0]:.0f}, {newest_person['center'][1]:.0f})")
                
                # Track the newest person (if still in frame, otherwise pick largest)
                if newest_person:
                    # Check if newest_person still exists by comparing bounding boxes
                    still_present = any(
                        abs(p['center'][0] - newest_person['center'][0]) < 50 and 
                        abs(p['center'][1] - newest_person['center'][1]) < 50
                        for p in persons
                    )
                    
                    if still_present:
                        # Update to latest position of the same person
                        for p in persons:
                            if abs(p['center'][0] - newest_person['center'][0]) < 50 and \
                               abs(p['center'][1] - newest_person['center'][1]) < 50:
                                newest_person = p
                                break
                    else:
                        # Newest person left, track largest remaining
                        newest_person = max(persons, key=lambda d: d["area"])
                        logger.info(f"🔄 Switching target to largest person | Area: {newest_person['area']:.0f}px²")
                
                # Center the camera on the newest person
                person_id = f"P{frame_count}" if not had_person else "TARGET"
                side, vert, debug_info = nudge_to_center(
                    servo, frame['w'], frame['h'], newest_person['bbox'], person_id
                )
                
                # Log summary every 30 frames if centered
                if frame_count % 30 == 0 and not debug_info['moved']:
                    logger.info(f"✓ TARGET LOCKED | Tracking {len(persons)} person(s) | Center offset: ({debug_info['offset_px'][0]:.0f}px, {debug_info['offset_px'][1]:.0f}px)")
                    
            else:
                if had_person:
                    logger.info("👋 ALL PERSONS LEFT FRAME")
                    newest_person = None
                had_person = False
                
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down EVE Security System...")
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}", exc_info=True)
    finally:
        servo_pos = servo.get_position()
        logger.info(f"📊 Final servo position: Pan={servo_pos['pan']}°, Tilt={servo_pos['tilt']}°")
        logger.info("👋 EVE Security System stopped")

def run_eve_system():
    """Run the complete EVE system with async support"""
    try:
        # Check if there's already an event loop running
        try:
            loop = asyncio.get_running_loop()
            logger.info("Using existing event loop")
        except RuntimeError:
            # No loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("Created new event loop")
        
        # Run the main detection loop
        main()
        
    except Exception as e:
        logger.error(f"Failed to start EVE system: {e}")
    finally:
        # Clean up only if we created the loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running() and not loop.is_closed():
                loop.close()
        except Exception:
            pass

if __name__ == "__main__":
    print("🚀 EVE (Enhanced Visual Enforcer) Security System")
    print("=" * 50)
    run_eve_system()