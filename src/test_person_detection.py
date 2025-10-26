#!/usr/bin/env python3
"""
Test script for person detection integration with EVE Security System
"""
import asyncio
import time
import logging
from yolo_backend.detector import YOLODetector
from notifications.person_detector import person_notifier
from adk_backend.agent import root_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_person_detection():
    """Test the person detection and notification system"""
    
    print("🧪 EVE Person Detection Test")
    print("=" * 40)
    
    # Test 1: Agent initialization
    print("\n1. Testing ADK Agent...")
    try:
        # Properly handle the async generator
        response_generator = root_agent.run_async("Initialize security monitoring mode and center the camera.")
        responses = []
        async for response in response_generator:
            responses.append(response)
        
        final_response = responses[-1] if responses else "No response"
        print(f"   ✅ Agent Response: {final_response}")
    except Exception as e:
        print(f"   ❌ Agent Error: {e}")
    
    # Test 2: Mock person detection event
    print("\n2. Testing Person Detection Notification...")
    
    # Simulate person detection event
    mock_event = {
        "event": "person_entered",
        "person_count": 1,
        "timestamp": time.time(),
        "persons": [{
            "cls": "person",
            "conf": 0.85,
            "bbox": [100, 50, 300, 400],
            "center": [200, 225],
            "area": 70000
        }]
    }
    
    try:
        person_notifier.on_person_detected(mock_event)
        print("   ✅ Person detection event processed")
        
        # Wait for agent response
        await asyncio.sleep(3)
        
    except Exception as e:
        print(f"   ❌ Person detection error: {e}")
    
    # Test 3: Person left event
    print("\n3. Testing Person Left Notification...")
    
    mock_left_event = {
        "event": "person_left",
        "person_count": 0,
        "timestamp": time.time(),
        "persons": []
    }
    
    try:
        person_notifier.on_person_detected(mock_left_event)
        print("   ✅ Person left event processed")
        
        # Wait for agent response
        await asyncio.sleep(3)
        
    except Exception as e:
        print(f"   ❌ Person left error: {e}")
    
    # Test 4: Check notification status
    print("\n4. Checking System Status...")
    try:
        status = person_notifier.get_status()
        print(f"   📊 Notifier Status: {status}")
        
        recent_events = person_notifier.get_recent_events(5)
        print(f"   📋 Recent Events: {len(recent_events)} events")
        
    except Exception as e:
        print(f"   ❌ Status check error: {e}")

def test_yolo_detector():
    """Test YOLO detector with callback registration"""
    print("\n5. Testing YOLO Detector Integration...")
    
    def test_callback(event_data):
        """Test callback function"""
        event_type = event_data.get("event")
        person_count = event_data.get("person_count", 0)
        print(f"   📨 Callback received: {event_type} with {person_count} person(s)")
    
    try:
        # Initialize detector (this will work even without camera)
        detector = YOLODetector(
            source="/dev/video0",  # This might fail without camera
            confidence_threshold=0.6
        )
        
        # Register callback
        detector.add_detection_callback(test_callback)
        print("   ✅ YOLO detector initialized and callback registered")
        
        # Check status
        status = detector.get_detection_status()
        print(f"   📊 Detector Status: {status}")
        
    except Exception as e:
        print(f"   ⚠️  YOLO detector test (expected on systems without camera): {e}")

def main():
    """Main test function"""
    print("🚀 Starting EVE Person Detection Tests...")
    
    # Test YOLO detector first (sync)
    test_yolo_detector()
    
    # Test async components
    try:
        asyncio.run(test_person_detection())
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    print("\n✨ Test completed!")
    print("\nTo run the full EVE system:")
    print("python main.py")

if __name__ == "__main__":
    main()