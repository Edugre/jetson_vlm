import time
import platform

# Try to import hardware libraries, use mocks if not available
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
except (ImportError, NotImplementedError):
    HARDWARE_AVAILABLE = False
    print("⚠ Hardware libraries not available - using mock mode")

    # Mock classes for testing without hardware
    class MockPCA9685:
        def __init__(self, i2c):
            self.frequency = 50
            self.channels = [MockChannel() for _ in range(16)]
            print("Mock PCA9685 initialized")

        def deinit(self):
            print("Mock PCA9685 deinitialized")

    class MockChannel:
        pass

    class MockServo:
        def __init__(self, channel):
            self._angle = 90
            print(f"Mock servo initialized on channel")

        @property
        def angle(self):
            return self._angle

        @angle.setter
        def angle(self, value):
            self._angle = value
            print(f"Mock servo angle set to: {value}°")

class ServoController:
    def __init__(self, pan_channel=0, tilt_channel=1):
        if HARDWARE_AVAILABLE:
            # Real hardware initialization
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = 50

            self.pan_servo = servo.Servo(self.pca.channels[pan_channel])
            self.tilt_servo = servo.Servo(self.pca.channels[tilt_channel])
            print("✓ Real hardware initialized")
        else:
            # Mock hardware initialization
            self.pca = MockPCA9685(None)
            self.pan_servo = MockServo(self.pca.channels[pan_channel])
            self.tilt_servo = MockServo(self.pca.channels[tilt_channel])
            print("✓ Mock hardware initialized")

        # Track current angles
        self.pan_angle = 90
        self.tilt_angle = 90

        # Initialize to center position
        self.pan_servo.angle = self.pan_angle
        self.tilt_servo.angle = self.tilt_angle
        time.sleep(0.5)
    
    def move_up(self, step=10):
        self.tilt_angle = max(0, self.tilt_angle - step)
        self.tilt_servo.angle = self.tilt_angle
        return {"direction": "up", "new_angle": self.tilt_angle}
    
    def move_down(self, step=10):
        self.tilt_angle = min(180, self.tilt_angle + step)
        self.tilt_servo.angle = self.tilt_angle
        return {"direction": "down", "new_angle": self.tilt_angle}
    
    def move_left(self, step=10):
        self.pan_angle = max(0, self.pan_angle - step)
        self.pan_servo.angle = self.pan_angle
        return {"direction": "left", "new_angle": self.pan_angle}
    
    def move_right(self, step=10):
        self.pan_angle = min(180, self.pan_angle + step)
        self.pan_servo.angle = self.pan_angle
        return {"direction": "right", "new_angle": self.pan_angle}
    
    def set_pan_tilt(self, pan_angle, tilt_angle):
        self.pan_angle = max(0, min(180, pan_angle))
        self.tilt_angle = max(0, min(180, tilt_angle))
        self.pan_servo.angle = self.pan_angle
        self.tilt_servo.angle = self.tilt_angle
        return {"pan": self.pan_angle, "tilt": self.tilt_angle}
    
    def center(self):
        return self.set_pan_tilt(90, 90)
    
    def get_position(self):
        return {"pan": self.pan_angle, "tilt": self.tilt_angle}
    
    def cleanup(self):
        """Cleanup resources. ServoKit handles cleanup automatically."""
        pass