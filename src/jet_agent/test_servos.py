from adafruit_servokit import ServoKit
import time

# Initialize PCA9685 with 16 channels
kit = ServoKit(channels=16)

# Define servo channels for X and Y axis
Y_AXIS_SERVO = kit.servo[0]  # Channel 0 - Horizontal movement
X_AXIS_SERVO = kit.servo[1]  # Channel 1 - Vertical movement

print("Testing X-Axis Servo (Channel 0)...")
# Test X-axis servo
X_AXIS_SERVO.angle = 0      # Move to 0 degrees (far left)
time.sleep(1)
X_AXIS_SERVO.angle = 90     # Move to 90 degrees (center)
time.sleep(1)
X_AXIS_SERVO.angle = 180    # Move to 180 degrees (far right)
time.sleep(1)
X_AXIS_SERVO.angle = 90     # Back to center
print("X-Axis test complete!")

print("\nTesting Y-Axis Servo (Channel 1)...")
# Test Y-axis servo
Y_AXIS_SERVO.angle = 0      # Move to 0 degrees (bottom)
time.sleep(1)
Y_AXIS_SERVO.angle = 90     # Move to 90 degrees (center)
time.sleep(1)
Y_AXIS_SERVO.angle = 180    # Move to 180 degrees (top)
time.sleep(1)
Y_AXIS_SERVO.angle = 90     # Back to center
print("Y-Axis test complete!")

print("\nBoth servos tested successfully!")