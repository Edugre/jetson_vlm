#!/usr/bin/env python3
"""
Simple test script for ServoController
Tests the pan/tilt servo movements
"""
import time
from src.jet_agent.servo_controller import ServoController

def main():
    print("Initializing Servo Controller...")
    try:
        controller = ServoController()
        print("✓ Servo Controller initialized successfully")
        print(f"Initial position: {controller.get_position()}\n")
        
        # Test centering
        print("Test 1: Centering servos...")
        controller.center()
        print(f"Position: {controller.get_position()}\n")
        time.sleep(1)
        
        # Test pan movements
        print("Test 2: Moving right...")
        result = controller.move_right(30)
        print(f"Result: {result}")
        time.sleep(1)
        
        print("Test 3: Moving left...")
        result = controller.move_left(30)
        print(f"Result: {result}")
        time.sleep(1)
        
        # Test tilt movements
        print("Test 4: Moving up...")
        result = controller.move_up(20)
        print(f"Result: {result}")
        time.sleep(1)
        
        print("Test 5: Moving down...")
        result = controller.move_down(20)
        print(f"Result: {result}")
        time.sleep(1)
        
        # Test specific position
        print("Test 6: Setting specific position (45, 135)...")
        result = controller.set_pan_tilt(45, 135)
        print(f"Result: {result}")
        time.sleep(1)
        
        # Return to center
        print("\nReturning to center position...")
        controller.center()
        print(f"Final position: {controller.get_position()}")
        
        # Cleanup
        print("\nCleaning up...")
        controller.cleanup()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("- You're running on a Jetson device")
        print("- I2C is enabled")
        print("- PCA9685 servo controller is connected")
        print("- Servos are connected to channels 0 (pan) and 1 (tilt)")

if __name__ == "__main__":
    main()
