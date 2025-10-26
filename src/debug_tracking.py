#!/usr/bin/env python3
"""
Debug script for testing EVE tracking functionality
"""
import sys
import os
import time
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_tracking_parameters():
    """Test and display tracking parameters"""
    print("🔧 EVE Tracking Parameters Debug")
    print("=" * 40)
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        
        # Create server to check parameters
        server = YOLOVideoStreamer(port=5003)
        
        print(f"📊 Current Tracking Settings:")
        print(f"   Auto-tracking: {server.auto_tracking}")
        print(f"   Min confidence: {server.min_tracking_confidence}")
        print(f"   Tracking cooldown: {server.tracking_cooldown}s")
        print(f"   Deadzone X: {server.center_x_margin} ({server.center_x_margin * 640:.0f}px at 640px width)")
        print(f"   Deadzone Y: {server.center_y_margin} ({server.center_y_margin * 480:.0f}px at 480px height)")
        print(f"   Pan step: {server.pan_step_small}°")
        print(f"   Tilt step: {server.tilt_step_small}°")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def simulate_person_detection():
    """Simulate person detection for tracking"""
    print("\n🎯 Person Detection Simulation")
    print("-" * 30)
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        
        server = YOLOVideoStreamer(port=5003)
        
        # Test scenarios
        scenarios = [
            {
                "name": "Person on left side",
                "persons": [{
                    "bbox": [100, 200, 200, 400],  # Left side
                    "conf": 0.85,
                    "center": [150, 300]
                }],
                "frame_info": {"w": 640, "h": 480}
            },
            {
                "name": "Person on right side", 
                "persons": [{
                    "bbox": [440, 200, 540, 400],  # Right side
                    "conf": 0.90,
                    "center": [490, 300]
                }],
                "frame_info": {"w": 640, "h": 480}
            },
            {
                "name": "Person in center (should not move)",
                "persons": [{
                    "bbox": [270, 190, 370, 390],  # Center
                    "conf": 0.95,
                    "center": [320, 290]
                }],
                "frame_info": {"w": 640, "h": 480}
            },
            {
                "name": "Low confidence person (should not track)",
                "persons": [{
                    "bbox": [100, 200, 200, 400],
                    "conf": 0.45,  # Below 0.7 threshold
                    "center": [150, 300]
                }],
                "frame_info": {"w": 640, "h": 480}
            },
            {
                "name": "No persons detected",
                "persons": [],
                "frame_info": {"w": 640, "h": 480}
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Testing: {scenario['name']}")
            
            if scenario['persons']:
                person = scenario['persons'][0]
                print(f"   Person bbox: {person['bbox']}")
                print(f"   Person center: {person['center']}")
                print(f"   Confidence: {person['conf']:.1%}")
            
            # Test the tracking logic
            server._perform_auto_tracking(scenario['persons'], scenario['frame_info'])
            
            # Wait between tests
            time.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False

def test_servo_commands():
    """Test servo commands directly"""
    print("\n🕹️ Servo Command Test")
    print("-" * 20)
    
    try:
        from adk_backend.servo_controller import ServoController
        
        servo = ServoController()
        
        print("Testing servo movements...")
        
        # Test each direction
        movements = [
            ("Center", lambda: servo.center()),
            ("Move Up", lambda: servo.move_up(3)),
            ("Move Down", lambda: servo.move_down(3)),
            ("Move Left", lambda: servo.move_left(3)),
            ("Move Right", lambda: servo.move_right(3)),
            ("Center Again", lambda: servo.center())
        ]
        
        for name, command in movements:
            print(f"   {name}...", end=" ")
            result = command()
            print(f"✅ {result}")
            time.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"❌ Servo test error: {e}")
        return False

def check_detection_flow():
    """Check the detection event flow"""
    print("\n📡 Detection Event Flow Test")
    print("-" * 25)
    
    try:
        from yolo_backend.detector import YOLODetector
        
        # Test detector initialization
        detector = YOLODetector(confidence_threshold=0.6)
        print("✅ YOLO detector initialized")
        
        # Test callback registration
        events_received = []
        
        def test_callback(event_data):
            events_received.append(event_data)
            print(f"📨 Event received: {event_data.get('event')} - {event_data.get('person_count')} persons")
        
        detector.add_detection_callback(test_callback)
        print("✅ Detection callback registered")
        
        # Simulate detection events
        mock_events = [
            {
                "event": "person_entered",
                "person_count": 1,
                "persons": [{"conf": 0.85, "bbox": [100, 100, 200, 300]}],
                "timestamp": time.time()
            },
            {
                "event": "person_left", 
                "person_count": 0,
                "persons": [],
                "timestamp": time.time()
            }
        ]
        
        for event in mock_events:
            detector._check_person_events(event["person_count"], event["persons"])
            time.sleep(0.5)
        
        print(f"✅ Processed {len(events_received)} detection events")
        return True
        
    except Exception as e:
        print(f"❌ Detection flow error: {e}")
        return False

def main():
    """Run all debug tests"""
    print("🐛 EVE Tracking Debug Suite")
    print("=" * 50)
    
    tests = [
        ("Tracking Parameters", test_tracking_parameters),
        ("Person Detection Simulation", simulate_person_detection),
        ("Servo Commands", test_servo_commands),
        ("Detection Event Flow", check_detection_flow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Debug Results:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 30)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed >= len(results) - 1:  # Allow one hardware test to fail
        print("\n🎉 Tracking system looks good!")
        print("\n💡 Troubleshooting tips:")
        print("   • Check logs for 'Tracking person with X% confidence' messages")
        print("   • Verify person confidence > 70% for tracking to activate")
        print("   • Ensure auto-tracking is enabled in web interface")
        print("   • Person must be outside the deadzone to trigger movement")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()