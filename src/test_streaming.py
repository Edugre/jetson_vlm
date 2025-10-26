#!/usr/bin/env python3
"""
Test script for the EVE video streaming system
"""
import sys
import os
import time
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("   ✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"   ❌ OpenCV import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("   ✅ Flask imported successfully")
    except ImportError as e:
        print(f"   ❌ Flask import failed: {e}")
        return False
    
    try:
        from flask_socketio import SocketIO
        print("   ✅ Flask-SocketIO imported successfully")
    except ImportError as e:
        print(f"   ❌ Flask-SocketIO import failed: {e}")
        return False
    
    try:
        from yolo_backend.detector import YOLODetector
        print("   ✅ YOLO Detector imported successfully")
    except ImportError as e:
        print(f"   ❌ YOLO Detector import failed: {e}")
        return False
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        print("   ✅ Video Streamer imported successfully")
    except ImportError as e:
        print(f"   ❌ Video Streamer import failed: {e}")
        return False
    
    return True

def test_camera_access():
    """Test camera access"""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)  # Try first camera
        
        if not cap.isOpened():
            print("   ⚠️  Camera not accessible (expected on systems without camera)")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"   ✅ Camera accessible - Frame size: {frame.shape}")
            return True
        else:
            print("   ❌ Failed to read frame from camera")
            return False
            
    except Exception as e:
        print(f"   ❌ Camera test error: {e}")
        return False

def test_yolo_detector():
    """Test YOLO detector initialization"""
    print("\n🎯 Testing YOLO detector...")
    
    try:
        from yolo_backend.detector import YOLODetector
        
        # Test detector initialization (might fail without camera)
        detector = YOLODetector(
            source="/dev/video0",
            confidence_threshold=0.6
        )
        
        print("   ✅ YOLO detector initialized successfully")
        
        # Test status
        status = detector.get_detection_status()
        print(f"   📊 Detector status: {status}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  YOLO detector test (may fail without camera): {e}")
        return False

def test_video_server():
    """Test video server initialization"""
    print("\n🌐 Testing video server...")
    
    try:
        from streaming.video_server import YOLOVideoStreamer
        
        # Create server instance (don't start it)
        server = YOLOVideoStreamer(
            camera_source="/dev/video0",
            confidence_threshold=0.6,
            port=5001  # Use different port for testing
        )
        
        print("   ✅ Video server initialized successfully")
        print(f"   📱 Server would be available at: http://localhost:5001")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Video server test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 EVE Video Streaming System - Test Suite")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Test", test_camera_access),
        ("YOLO Test", test_yolo_detector),
        ("Server Test", test_video_server)
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
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 30)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to start streaming:")
        print("   python stream_app.py")
    else:
        print("\n⚠️  Some tests failed. Check dependencies:")
        print("   pip install Flask Flask-SocketIO")
        
        if not any(result for name, result in results if "Camera" in name):
            print("   📹 Camera not detected - streaming will work with any video source")

if __name__ == "__main__":
    main()