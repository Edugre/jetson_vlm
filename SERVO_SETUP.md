# Servo Controller Setup - Complete

## Overview
The Jetson agent is now fully integrated with servo control using the `adafruit_servokit` library.

## Hardware Configuration
- **Channel 0 (X-axis)**: Pan servo - Horizontal movement (left/right)
- **Channel 1 (Y-axis)**: Tilt servo - Vertical movement (up/down)

## Files Updated

### 1. `servo_controller.py`
Updated to use `adafruit_servokit` library instead of the lower-level `adafruit_pca9685`.
- Simpler, more reliable API
- Better compatibility with Jetson platform
- Consistent with test scripts

### 2. `agent.py`
Already configured with 8 tools for camera control:
1. `get_current_time` - Time queries
2. `move_camera_up` - Tilt up
3. `move_camera_down` - Tilt down
4. `move_camera_left` - Pan left
5. `move_camera_right` - Pan right
6. `center_camera` - Reset to center (90°, 90°)
7. `set_camera_position` - Set specific angles
8. `get_camera_position` - Get current angles

## Test Scripts

### Basic Servo Test
```bash
cd src/jet_agent
python3 test_servos.py
```
Tests raw servo movement on both channels.

### Servo Controller Test
```bash
cd src/jet_agent
python3 test_agent_servos.py
```
Tests the ServoController class methods.

### Full Agent Integration Test
```bash
cd src
python3 test_agent_integration.py
```
Verifies the complete agent setup with servo control.

## Usage Example

```python
from jet_agent.agent import root_agent, servo_controller

# Direct servo control
servo_controller.move_right(15)
servo_controller.move_up(10)
servo_controller.center()

# Or use the agent's tools
# The agent can control the servos through natural language commands
```

## Movement Ranges
- **Pan (X-axis)**: 0° - 180° (left to right)
- **Tilt (Y-axis)**: 0° - 180° (bottom to top)
- **Default center**: 90°, 90°

## Notes
- No sudo required for servo control (user has I2C permissions)
- ServoKit automatically handles cleanup
- Safe angle limiting prevents servo damage
