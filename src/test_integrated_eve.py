#!/usr/bin/env python3
"""
Test script for the integrated EVE streaming system
"""
import sys
import os
import time
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_integrated_imports():
    """Test all integrated system imports"""
    print("🧪 Testing integrated EVE system imports...")
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        print("   ✅ Video Streamer imported")
    except ImportError as e:
        print(f"   ❌ Video Streamer import failed: {e}")
        return False
    
    try:
        from yolo_backend.detector import YOLODetector
        print("   ✅ YOLO Detector imported")
    except ImportError as e:
        print(f"   ❌ YOLO Detector import failed: {e}")
        return False
    
    try:
        from adk_backend.servo_controller import ServoController
        print("   ✅ Servo Controller imported")
    except ImportError as e:
        print(f"   ❌ Servo Controller import failed: {e}")
        return False
    
    try:
        from notifications.person_detector import person_notifier
        print("   ✅ Person Notifier imported")
    except ImportError as e:
        print(f"   ❌ Person Notifier import failed: {e}")
        return False
    
    return True

def test_servo_integration():
    """Test servo controller integration"""
    print("\n🕹️ Testing servo controller integration...")
    
    try:
        from adk_backend.servo_controller import ServoController
        
        servo = ServoController()
        print("   ✅ Servo controller initialized")
        
        # Test basic movements
        center_result = servo.center()
        print(f"   🎯 Center test: {center_result}")
        
        up_result = servo.move_up(5)
        print(f"   ⬆️  Move up test: {up_result}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Servo test (may fail without hardware): {e}")
        return False

def test_integrated_server():
    """Test integrated streaming server"""
    print("\n🌐 Testing integrated streaming server...")
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        
        # Create server instance (don't start it)
        server = YOLOVideoStreamer(
            camera_source="/dev/video0",
            confidence_threshold=0.6,
            port=5002  # Use different port for testing
        )
        
        print("   ✅ Integrated server initialized")
        print(f"   📱 Would be available at: http://localhost:5002")
        
        # Check if servo controller is integrated
        if hasattr(server, 'servo_controller'):
            print("   ✅ Servo controller integrated")
        else:
            print("   ❌ Servo controller not integrated")
            return False
        
        # Check if auto-tracking is available
        if hasattr(server, 'auto_tracking'):
            print("   ✅ Auto-tracking available")
        else:
            print("   ❌ Auto-tracking not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated server test failed: {e}")
        return False

def test_detection_integration():
    """Test person detection integration"""
    print("\n🎯 Testing person detection integration...")
    
    try:
        from yolo_backend.detector import YOLODetector
        from notifications.person_detector import person_notifier
        
        # Test detector
        detector = YOLODetector(confidence_threshold=0.6)
        print("   ✅ YOLO detector initialized")
        
        # Test callback system
        def test_callback(event_data):
            print(f"   📨 Test callback received: {event_data.get('event')}")
        
        detector.add_detection_callback(test_callback)
        print("   ✅ Detection callback registered")
        
        # Test person notifier
        status = person_notifier.get_status()
        print(f"   📊 Person notifier status: {status}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Detection integration test: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions for the integrated system"""
    print("\n🚀 EVE Integrated System Usage Instructions")
    print("=" * 50)
    
    print("\n📋 HOW TO USE:")
    print("1. Start the integrated system:")
    print("   python stream_app.py")
    print("")
    print("2. Open your web browser to:")
    print("   http://localhost:5000")
    print("")
    print("3. Use the web interface:")
    print("   • Click 'Start Stream' to begin video")
    print("   • Toggle 'Auto-Track' for automatic person following")
    print("   • Use manual controls to move camera")
    print("   • Watch the log for detection events")
    print("")
    print("🎯 FEATURES YOU'LL SEE:")
    print("   • Live video with person detection overlays")
    print("   • Automatic camera movement when persons detected")
    print("   • Security alerts in the web interface")
    print("   • Manual camera control with arrow buttons")
    print("   • Real-time statistics and detection logs")
    print("")
    print("🤖 EVE BEHAVIORS:")
    print("   • Person enters → Camera automatically tracks")
    print("   • Person leaves → Camera returns to center")
    print("   • Multiple persons → Tracks most confident detection")
    print("   • Manual override → Use controls anytime")

def main():
    """Run all tests"""
    print("🚀 EVE Integrated System - Test Suite")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Integrated Imports", test_integrated_imports),
        ("Servo Integration", test_servo_integration),
        ("Integrated Server", test_integrated_server),
        ("Detection Integration", test_detection_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Summary:")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 30)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed >= 3:  # Allow for some hardware-dependent failures
        print("\n🎉 EVE Integrated System ready!")
        show_usage_instructions()
    else:
        print("\n⚠️  Some critical tests failed. Check dependencies.")
        print("   pip install Flask Flask-SocketIO")

if __name__ == "__main__":
    main()