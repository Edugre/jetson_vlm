#!/usr/bin/env python3
"""Test script to verify servo_controller integration."""

from servo_controller import ServoController
import time

def test_servo_controller():
    """Test all servo controller functions."""
    print("Initializing ServoController...")
    controller = ServoController()

    print("\n1. Testing get_position():")
    pos = controller.get_position()
    print(f"   Current position: {pos}")

    print("\n2. Testing move_right():")
    result = controller.move_right(15)
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n3. Testing move_left():")
    result = controller.move_left(15)
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n4. Testing move_up():")
    result = controller.move_up(15)
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n5. Testing move_down():")
    result = controller.move_down(15)
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n6. Testing set_pan_tilt(45, 135):")
    result = controller.set_pan_tilt(45, 135)
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n7. Testing center():")
    result = controller.center()
    print(f"   Result: {result}")
    time.sleep(1)

    print("\n8. Final position:")
    pos = controller.get_position()
    print(f"   Position: {pos}")

    print("\n✓ All tests completed successfully!")
    controller.cleanup()

if __name__ == "__main__":
    test_servo_controller()
