# EVE Security System 

**EVE** (Enhanced Visual Enforcer) is an intelligent security camera system that actively tracks and monitors anomalies in real-time. Using advanced computer vision and AI, EVE can follow suspicious activities, respond to natural language commands, and autonomously optimize its viewing angle to capture the most relevant activity in a room.

## 🆕 NEW: Live Video Streaming Frontend

EVE now includes a web-based live streaming interface with real-time person detection overlays!

### Quick Start - Video Streaming

```bash
# Install web dependencies
pip install Flask Flask-SocketIO

# Start the live video stream
cd src
python stream_app.py

# Open your browser to: http://localhost:5000
```

**Features:**
- 🎥 **Live video streaming** with YOLO person detection
- 🎯 **Real-time detection overlays** (bounding boxes, confidence scores)
- 📊 **Live statistics** (FPS, person count, detection history)
- 🚨 **Security alerts** when persons enter/leave frame
- 📱 **Responsive web interface** works on desktop and mobile
- 🤖 **Integrated with EVE agent** for automatic camera tracking

## Features

### Intelligent Tracking & Monitoring
- **Real-time Anomaly Detection**: Detects and tracks suspicious activities including:
  - Object theft or removal from the room
  - Physical altercations between people
  - Unauthorized access or unusual behavior
- **Multi-person Tracking**: Uses YOLO for robust person detection and tracking
- **Autonomous Frame Optimization**: Automatically adjusts camera angle to capture the maximum number of people and anomalies
- **Live Web Streaming**: Real-time video stream with detection overlays

### AI-Powered Interaction
- **Natural Language Commands**: Responds to voice/text commands such as:
  - *"Make sure you don't take your eyes off of this person as long as they are on the frame"*
  - *"Track everyone in the room"*
  - *"Focus on the person near the door"*
- **Google ADK Integration**: Powered by Gemini AI in the cloud for intelligent decision-making
- **Context-Aware Responses**: Understands and executes complex monitoring instructions
- **Automatic Tracking**: When persons are detected, EVE automatically adjusts camera position

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

- **Computer Vision**: YOLO for object detection and tracking
- **AI/ML**: Google ADK with Gemini 2.5 Flash model
- **Web Interface**: Flask + SocketIO for live streaming
- **Hardware Interface**: Adafruit libraries for servo control
- **Platform**: Python 3.10+

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

### 🎥 Live Video Streaming (NEW!)

```bash
# Start the web streaming interface
cd src
python stream_app.py

# Open browser to: http://localhost:5000
# Or access remotely: http://YOUR_JETSON_IP:5000
```

**Streaming Options:**
```bash
# Custom camera source
python stream_app.py --camera /dev/video1

# Custom port
python stream_app.py --port 8080

# Adjust detection sensitivity
python stream_app.py --confidence 0.7

# Debug mode
python stream_app.py --debug
```

### 🤖 Running EVE Security System

```bash
# Run the complete EVE system with person detection
python main.py
```

### 🧪 Testing Components

```bash
# Test person detection integration
python test_person_detection.py

# Test video streaming components
python test_streaming.py

# Test servo control
python servo_tools/test_servo.py
```

### Example Commands
Once EVE is running, you can give it commands like:
- *"Track the person in the blue shirt"*
- *"Monitor the door for anyone entering"*
- *"Keep an eye on everyone in the frame"*
- *"Alert me if someone takes something"*

## 📱 Web Interface Features

The new web interface provides:

### 🎯 Real-time Detection
- Live video stream with person detection overlays
- Bounding boxes around detected persons
- Confidence scores and position coordinates
- Center point markers for tracking

### 📊 Live Statistics
- **FPS**: Real-time frame rate
- **Persons in Frame**: Current person count
- **Frames Processed**: Total frames analyzed
- **Last Detection**: Timestamp of last person detection

### 🚨 Security Alerts
- Visual alerts when persons enter/leave the monitoring area
- Real-time person count changes
- Integration with EVE's intelligent tracking system

### 📝 System Log
- Real-time event logging
- Detection events and system status
- Timestamped entries with color coding

## Project Structure

```
jetson_vlm/
├── src/
│   ├── adk_backend/           # Google ADK agent integration
│   │   ├── agent.py           # EVE Security Agent with camera tools
│   │   └── servo_controller.py # Servo control with mock support
│   ├── yolo_backend/          # YOLO detection system
│   │   └── detector.py        # Enhanced YOLO detector with events
│   ├── notifications/         # Person detection notifications
│   │   └── person_detector.py # ADK agent notification system
│   ├── streaming/             # NEW: Web streaming interface
│   │   ├── video_server.py    # Flask/SocketIO streaming server
│   │   └── templates/
│   │       └── index.html     # Web interface
│   ├── utils/
│   │   └── logger.py          # Logging utilities
│   ├── main.py                # Main EVE system
│   ├── stream_app.py          # NEW: Streaming application launcher
│   └── test_*.py              # Test scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Development & Testing

### Mock Mode for Development
The servo controller automatically detects if it's running on hardware without I2C support (like Windows) and switches to mock mode. This allows you to:
- Develop and test logic on any machine
- Verify servo movements without physical hardware
- Seamlessly deploy to Jetson without code changes

When deployed to the Jetson with proper hardware, EVE automatically switches to real hardware mode.

### Testing the System

```bash
# Test all components
python test_streaming.py

# Test person detection
python test_person_detection.py

# Test with mock camera (for development)
python stream_app.py --camera 0
```

## How It Works

1. **Video Capture**: Camera captures live video feed
2. **Person Detection**: YOLO identifies and tracks people in the frame
3. **Event Processing**: Detection events trigger notifications
4. **AI Decision Making**: EVE agent processes alerts and decides camera positioning
5. **Servo Control**: Pan/tilt servos adjust camera angle to track persons
6. **Web Streaming**: Live video with overlays streams to web interface
7. **Continuous Monitoring**: System loops, constantly adapting to activity

## API Endpoints

The streaming server provides these endpoints:

- `GET /` - Main web interface
- `GET /stream` - Live video stream (MJPEG)
- `GET /api/stats` - Get streaming statistics
- `GET /api/start_stream` - Start video streaming
- `GET /api/stop_stream` - Stop video streaming
- `GET /api/detection_status` - Get detection status

## Security & Privacy

EVE is designed for authorized security monitoring only. Ensure compliance with local privacy laws and regulations when deploying this system.

## Troubleshooting

### Camera Issues
```bash
# List available cameras
ls /dev/video*

# Test camera access
python -c "import cv2; print('Camera works!' if cv2.VideoCapture(0).isOpened() else 'Camera not found')"
```

### Web Interface Not Loading
```bash
# Check if server is running
curl http://localhost:5000/api/stats

# Try different port
python stream_app.py --port 8080
```

### Permission Issues
```bash
# Add user to video group (may require logout/login)
sudo usermod -a -G video $USER
```

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
- Ultralytics for YOLO implementation
- NVIDIA for the Jetson platform

---

**Note**: This system is actively maintained and includes both the core security functionality and the new web streaming interface.