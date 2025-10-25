from google.adk.agents.llm_agent import Agent
from .servo_controller import ServoController

servo_controller = ServoController()

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

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description="Controls camera servos and tells time.",
    instruction="You can control a camera mounted on pan/tilt servos and tell the current time. Use servo movement tools to position the camera and get_current_time for time queries.",
    tools=[
        get_current_time,
        move_camera_up,
        move_camera_down, 
        move_camera_left,
        move_camera_right,
        center_camera,
        set_camera_position,
        get_camera_position
    ],
)