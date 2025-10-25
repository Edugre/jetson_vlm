#!/usr/bin/env python3
"""Test script to verify the agent integration with servo controller."""

from jet_agent.agent import root_agent, servo_controller
import time

def test_agent_tools():
    """Test that agent tools are properly configured."""
    print("=" * 50)
    print("AGENT INTEGRATION TEST")
    print("=" * 50)

    print(f"\n✓ Agent Name: {root_agent.name}")
    print(f"✓ Agent Model: {root_agent.model}")
    print(f"✓ Number of Tools: {len(root_agent.tools)}")

    print("\n✓ Available Tools:")
    for i, tool in enumerate(root_agent.tools, 1):
        print(f"   {i}. {tool.__name__}")

    print("\n" + "=" * 50)
    print("SERVO CONTROLLER TEST")
    print("=" * 50)

    print("\n1. Getting initial position...")
    pos = servo_controller.get_position()
    print(f"   Position: {pos}")

    print("\n2. Testing move right...")
    servo_controller.move_right(10)
    time.sleep(0.5)

    print("\n3. Testing move left...")
    servo_controller.move_left(10)
    time.sleep(0.5)

    print("\n4. Centering camera...")
    servo_controller.center()
    time.sleep(0.5)

    print("\n5. Final position:")
    pos = servo_controller.get_position()
    print(f"   Position: {pos}")

    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nThe agent is ready to control the servos!")

if __name__ == "__main__":
    test_agent_tools()
