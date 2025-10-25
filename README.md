# EVE Security System 

**EVE** (Enhanced Visual Enforcer) is an intelligent security camera system that actively tracks and monitors anomalies in real-time. Using advanced computer vision and AI, EVE can follow suspicious activities, respond to natural language commands, and autonomously optimize its viewing angle to capture the most relevant activity in a room.

## Features

### Intelligent Tracking & Monitoring
- **Real-time Anomaly Detection**: Detects and tracks suspicious activities including:
  - Object theft or removal from the room
  - Physical altercations between people
  - Unauthorized access or unusual behavior
- **Multi-person Tracking**: Uses ByteTrack algorithm for robust person detection and tracking
- **Autonomous Frame Optimization**: Automatically adjusts camera angle to capture the maximum number of people and anomalies

### AI-Powered Interaction
- **Natural Language Commands**: Responds to voice/text commands such as:
  - *"Make sure you don't take your eyes off of this person as long as they are on the frame"*
  - *"Track everyone in the room"*
  - *"Focus on the person near the door"*
- **Google ADK Integration**: Powered by Gemini AI in the cloud for intelligent decision-making
- **Context-Aware Responses**: Understands and executes complex monitoring instructions

### Hardware Control
- **Pan/Tilt Servo System**: Motorized camera movement for 180° horizontal and vertical coverage
- **Raspberry Pi Camera Module 2**: High-quality video capture for person recognition
- **Smooth Tracking**: Precise servo control for seamless target following

## Hardware Requirements

- **NVIDIA Jetson Orin Nano** - Main processing unit
- **Raspberry Pi Camera Module 2** - Video capture
- **PCA9685 Servo Controller** - I2C servo driver board
- **2x Servo Motors** - Pan (horizontal) and tilt (vertical) movement
- **Power Supply** - Adequate power for Jetson and servos

## Software Stack

- **Computer Vision**: ByteTrack for object tracking
- **AI/ML**: Google ADK with Gemini 2.5 Flash model
- **Hardware Interface**: Adafruit libraries for servo control
- **Platform**: Python 3.11+

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Edugre/jetson_vlm.git
cd jetson_vlm
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Jetson
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Hardware Setup
1. Connect the PCA9685 servo controller to the Jetson's I2C pins
2. Connect pan servo to channel 0, tilt servo to channel 1
3. Mount the Raspberry Pi Camera Module 2 to the Jetson
4. Ensure I2C is enabled on the Jetson:
   ```bash
   sudo i2cdetect -y -r 1
   ```

## Usage

### Running EVE
```bash
python main.py  # (Coming soon)
```

### Testing Servo Control
```bash
python test_servo.py
```

### Example Commands
Once EVE is running, you can give it commands like:
- *"Track the person in the blue shirt"*
- *"Monitor the door for anyone entering"*
- *"Keep an eye on everyone in the frame"*
- *"Alert me if someone takes something"*

## Project Structure

```
jetson_vlm/
├── src/
│   └── jet_agent/
│       ├── __init__.py
│       ├── agent.py              # Google ADK agent with Gemini
│       └── servo_controller.py   # Servo control with mock support
├── test_servo.py                 # Servo testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Development & Testing

### Mock Mode for Development
The servo controller automatically detects if it's running on hardware without I2C support (like Windows) and switches to mock mode. This allows you to:
- Develop and test logic on any machine
- Verify servo movements without physical hardware
- Seamlessly deploy to Jetson without code changes

When deployed to the Jetson with proper hardware, EVE automatically switches to real hardware mode.

## How It Works

1. **Video Capture**: Raspberry Pi Camera captures live video feed
2. **Person Detection**: ByteTrack identifies and tracks people in the frame
3. **Anomaly Detection**: AI analyzes behavior for suspicious activities
4. **Decision Making**: Gemini AI processes natural language commands and decides camera positioning
5. **Servo Control**: Pan/tilt servos adjust camera angle to optimize view
6. **Continuous Monitoring**: System loops, constantly adapting to room activity

## Security & Privacy

EVE is designed for authorized security monitoring only. Ensure compliance with local privacy laws and regulations when deploying this system.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Jorge Taban**
- **Eduardo Goncalvez**
- **Alex Waisman**

## Acknowledgments

- Google ADK team for the agent framework
- Adafruit for excellent hardware libraries
- ByteTrack team for robust tracking algorithms
- NVIDIA for the Jetson platform

---

**Note**: This system is currently in active development. Some features may be incomplete or experimental.