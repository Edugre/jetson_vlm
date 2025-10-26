#!/usr/bin/env python3
"""
Test smooth tracking specifically for Y-axis issues
"""
import sys
import os
import time
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup detailed logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_y_axis_tracking():
    """Test Y-axis tracking specifically"""
    print("🔍 Y-Axis Tracking Debug Test")
    print("=" * 40)
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        
        # Create server with debug settings
        server = YOLOVideoStreamer(port=5004)
        
        print(f"📊 Tracking Parameters:")
        print(f"   Y-axis margin: {server.center_y_margin}")
        print(f"   Y-axis threshold: {server.center_y_margin * 2}")
        print(f"   Max step size: {server.max_step_size}")
        print(f"   Min confidence: {server.min_tracking_confidence}")
        
        # Test scenarios focused on Y-axis
        test_cases = [
            {
                "name": "Person at TOP of frame (should move UP)",
                "bbox": [300, 50, 400, 150],  # Top area
                "expected": "up"
            },
            {
                "name": "Person at BOTTOM of frame (should move DOWN)", 
                "bbox": [300, 350, 400, 450],  # Bottom area
                "expected": "down"
            },
            {
                "name": "Person ABOVE center (should move UP)",
                "bbox": [300, 150, 400, 250],  # Above center
                "expected": "up"
            },
            {
                "name": "Person BELOW center (should move DOWN)",
                "bbox": [300, 280, 400, 380],  # Below center
                "expected": "down"
            },
            {
                "name": "Person in Y-CENTER (should NOT move)",
                "bbox": [300, 215, 400, 265],  # Very centered
                "expected": "none"
            }
        ]
        
        frame_w, frame_h = 640, 480
        center_y = frame_h / 2  # 240
        
        print(f"\n🎯 Frame center Y: {center_y}")
        print(f"🎯 Y threshold: {frame_h * server.center_y_margin} pixels")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. {case['name']}")
            
            bbox = case['bbox']
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) / 2.0
            
            print(f"   Bbox: {bbox}")
            print(f"   Person Y center: {cy}")
            print(f"   Distance from center: {cy - center_y}")
            print(f"   Expected movement: {case['expected']}")
            
            # Create mock person data
            person_data = {
                "bbox": bbox,
                "conf": 0.85,
                "center": [(x1 + x2) / 2, cy]
            }
            
            # Test the tracking
            print(f"   Testing tracking...")
            moved = server._nudge_to_center(bbox, frame_w, frame_h, person_data)
            print(f"   Result: {'MOVED' if moved else 'NO MOVEMENT'}")
            
            time.sleep(1)  # Wait between tests
        
        return True
        
    except Exception as e:
        print(f"❌ Y-axis test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_y_movements():
    """Test servo Y movements directly"""
    print("\n🕹️ Direct Servo Y-Movement Test")
    print("-" * 30)
    
    try:
        from adk_backend.servo_controller import ServoController
        
        servo = ServoController()
        
        print("Testing Y-axis servo movements...")
        
        # Test Y movements specifically
        movements = [
            ("Get current position", lambda: servo.get_position()),
            ("Move UP 2 degrees", lambda: servo.move_up(2)),
            ("Move DOWN 4 degrees", lambda: servo.move_down(4)),
            ("Move UP 2 degrees (return)", lambda: servo.move_up(2)),
            ("Center camera", lambda: servo.center())
        ]
        
        for name, command in movements:
            print(f"   {name}...", end=" ")
            result = command()
            print(f"Result: {result}")
            time.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"❌ Servo Y-movement test error: {e}")
        return False

def analyze_detection_bboxes():
    """Analyze typical detection bounding boxes"""
    print("\n📐 Detection Bbox Analysis")
    print("-" * 25)
    
    # Simulate typical person detections
    typical_detections = [
        {"name": "Standing person", "bbox": [250, 100, 390, 450]},
        {"name": "Sitting person", "bbox": [200, 250, 440, 450]},
        {"name": "Person upper body", "bbox": [220, 80, 420, 300]},
        {"name": "Person far away", "bbox": [300, 200, 340, 280]},
        {"name": "Person close up", "bbox": [50, 50, 590, 430]}
    ]
    
    frame_w, frame_h = 640, 480
    center_x, center_y = frame_w/2, frame_h/2
    
    print(f"Frame: {frame_w}x{frame_h}, Center: ({center_x}, {center_y})")
    
    for detection in typical_detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        offset_x = (cx - center_x) / (frame_w / 2)
        offset_y = (cy - center_y) / (frame_h / 2)
        
        print(f"\n{detection['name']}:")
        print(f"   Bbox: {bbox}")
        print(f"   Center: ({cx:.0f}, {cy:.0f})")
        print(f"   Offsets: X={offset_x:.3f}, Y={offset_y:.3f}")
        
        # Check if it would trigger movement
        x_threshold = 0.08 * 2  # Default threshold
        y_threshold = 0.08 * 2
        
        would_move_x = abs(offset_x) > x_threshold
        would_move_y = abs(offset_y) > y_threshold
        
        print(f"   Would move X: {would_move_x} ({'RIGHT' if offset_x > 0 else 'LEFT'} if moving)")
        print(f"   Would move Y: {would_move_y} ({'DOWN' if offset_y > 0 else 'UP'} if moving)")

def main():
    """Run smooth tracking tests"""
    print("🎯 Smooth Tracking Debug Suite")
    print("=" * 50)
    
    tests = [
        ("Y-Axis Tracking", test_y_axis_tracking),
        ("Servo Y-Movements", test_servo_y_movements),
        ("Detection Analysis", analyze_detection_bboxes)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            test_func()
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
    
    print("\n💡 Debugging Tips:")
    print("   • Run with --debug to see detailed movement logs")
    print("   • Check web interface log for tracking messages")
    print("   • Y-axis should move when person is above/below center")
    print("   • Look for 'Y-axis check' debug messages in logs")

if __name__ == "__main__":
    main()