from google.adk.agents.llm_agent import Agent
from .servo_controller import ServoController
import logging

servo_controller = ServoController()
logger = logging.getLogger(__name__)

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

def move_camera_up(step: int = 10) -> dict:
    """Move the camera up by tilting servo."""
    print("Moved camera up")
    return servo_controller.move_up(step)

def move_camera_down(step: int = 10) -> dict:
    """Move the camera down by tilting servo."""
    print("Moved camera down")
    return servo_controller.move_down(step)

def move_camera_left(step: int = 10) -> dict:
    """Move the camera left by panning servo."""
    print("Moved camera left")
    return servo_controller.move_left(step)

def move_camera_right(step: int = 10) -> dict:
    """Move the camera right by panning servo."""
    print("Moved camera right")
    return servo_controller.move_right(step)

def center_camera() -> dict:
    """Center the camera to default position (pan=90, tilt=90)."""
    print("Center camera")
    return servo_controller.center()

def set_camera_position(pan_angle: int, tilt_angle: int) -> dict:
    """Set specific pan and tilt angles for the camera."""
    print(f'Moved camera to {tilt_angle} and {pan_angle}')
    return servo_controller.set_pan_tilt(pan_angle, tilt_angle)

def get_camera_position() -> dict:
    """Get current camera pan and tilt position."""
    return servo_controller.get_position()

def track_person_at_position(x: float, y: float, frame_width: int = 640, frame_height: int = 480) -> dict:
    """
    Track a person at specific screen coordinates by adjusting camera position.
    
    Args:
        x: Horizontal position (0 to frame_width)
        y: Vertical position (0 to frame_height)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    """
    logger.info(f"Tracking person at position ({x}, {y})")
    
    # Calculate center offset
    center_x = frame_width / 2
    center_y = frame_height / 2
    
    offset_x = x - center_x
    offset_y = y - center_y
    
    # Convert to servo movement (rough estimation)
    # Adjust these values based on your camera's field of view
    pan_adjustment = int(offset_x * 0.1)  # Scale factor for pan
    tilt_adjustment = int(offset_y * 0.1)  # Scale factor for tilt
    
    # Move camera towards target
    result = {"status": "tracking", "target": {"x": x, "y": y}}
    
    if abs(pan_adjustment) > 2:  # Dead zone
        if pan_adjustment > 0:
            move_result = servo_controller.move_right(abs(pan_adjustment))
            result["pan_action"] = f"moved_right_{abs(pan_adjustment)}"
        else:
            move_result = servo_controller.move_left(abs(pan_adjustment))
            result["pan_action"] = f"moved_left_{abs(pan_adjustment)}"
    else:
        result["pan_action"] = "centered"
    
    if abs(tilt_adjustment) > 2:  # Dead zone
        if tilt_adjustment > 0:
            move_result = servo_controller.move_down(abs(tilt_adjustment))
            result["tilt_action"] = f"moved_down_{abs(tilt_adjustment)}"
        else:
            move_result = servo_controller.move_up(abs(tilt_adjustment))
            result["tilt_action"] = f"moved_up_{abs(tilt_adjustment)}"
    else:
        result["tilt_action"] = "centered"
    
    return result

def initiate_security_scan() -> dict:
    """Initiate a security scan by moving camera to scan the area."""
    logger.info("Initiating security scan")
    
    # Return to center first
    center_result = servo_controller.center()
    
    return {
        "status": "security_scan_initiated",
        "message": "Camera centered and ready for security monitoring",
        "center_result": center_result
    }

def set_alert_mode(enabled: bool = True) -> dict:
    """Set the camera to high alert monitoring mode."""
    logger.info(f"Alert mode {'enabled' if enabled else 'disabled'}")
    
    if enabled:
        # Center camera for optimal monitoring
        servo_controller.center()
        return {
            "status": "alert_mode_enabled",
            "message": "Camera positioned for maximum surveillance coverage"
        }
    else:
        return {
            "status": "alert_mode_disabled", 
            "message": "Returning to normal operation mode"
        }

root_agent = Agent(
    model='gemini-2.5-flash',
    name='EVE_Security_Agent',
    description="EVE (Enhanced Visual Enforcer) - An intelligent security camera system that tracks and monitors anomalies in real-time.",
    instruction="""You are EVE, an intelligent security camera system. Your primary role is to:

1. SECURITY MONITORING: When alerted about person detection, immediately assess the situation and take appropriate action
2. CAMERA CONTROL: Use servo movements to track persons and optimize viewing angles
3. THREAT ASSESSMENT: Analyze person behavior and respond with appropriate security measures
4. AUTONOMOUS TRACKING: When a person enters the frame, use track_person_at_position() to follow them

Key behaviors:
- When notified of person entry, immediately acknowledge and begin tracking
- Use track_person_at_position() with the person's coordinates to follow them
- Maintain visual contact with all detected persons
- If multiple persons are detected, prioritize tracking based on proximity or suspicious behavior
- Use set_alert_mode(True) when security threats are detected
- Provide clear status updates about your actions

You have the ability to control camera positioning and should actively track detected persons.""",
    tools=[
        get_current_time,
        move_camera_up,
        move_camera_down, 
        move_camera_left,
        move_camera_right,
        center_camera,
        set_camera_position,
        get_camera_position,
        track_person_at_position,
        initiate_security_scan,
        set_alert_mode
    ],
)