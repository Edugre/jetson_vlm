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

async def enhanced_person_callback(event_data):
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
        
        # Forward to the person notifier
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
    
    had_person = False

    try:
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
                
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down EVE Security System...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        logger.info("👋 EVE Security System stopped")

def run_eve_system():
    """Run the complete EVE system with async support"""
    try:
        # Initialize asyncio event loop for ADK agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the main detection loop
        main()
        
    except Exception as e:
        logger.error(f"Failed to start EVE system: {e}")
    finally:
        # Clean up
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except:
            pass

if __name__ == "__main__":
    print("🚀 EVE (Enhanced Visual Enforcer) Security System")
    print("=" * 50)
    run_eve_system()