"""
Live video streaming server with YOLO person detection overlays
"""
import cv2
import asyncio
import logging
import time
import base64
import json
from typing import Dict, Any, Optional
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import threading
import numpy as np

from yolo_backend.detector import YOLODetector
from notifications.person_detector import person_notifier
from adk_backend.servo_controller import ServoController
from adk_backend.agent import (
    center_camera, set_camera_position, get_camera_position,
    track_person_at_position, initiate_security_scan, set_alert_mode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOVideoStreamer:
    """Video streaming server with YOLO person detection"""
    
    def __init__(self, 
                 camera_source="/dev/video0",
                 confidence_threshold=0.6,
                 host="0.0.0.0",
                 port=5000):
        """
        Initialize the video streaming server
        
        Args:
            camera_source: Camera device path
            confidence_threshold: YOLO confidence threshold
            host: Server host
            port: Server port
        """
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        self.host = host
        self.port = port
        
        # Flask app setup
        self.app = Flask(__name__, template_folder='templates')
        self.app.config['SECRET_KEY'] = 'eve_security_system'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # YOLO detector
        self.detector = None
        self.current_frame = None
        self.current_detections = []
        self.frame_lock = threading.Lock()
        
        # Servo controller for camera tracking
        self.servo_controller = ServoController()
        
        # Streaming control
        self.streaming = False
        self.stream_thread = None
        
        # EVE tracking parameters
        self.tracking_enabled = True
        self.auto_tracking = True
        self.pan_step_small = 3
        self.tilt_step_small = 3
        self.center_x_margin = 0.15  # Larger deadzone to prevent jitter
        self.center_y_margin = 0.15
        
        # Tracking control
        self.last_tracking_time = 0
        self.tracking_cooldown = 0.5  # Minimum time between movements
        self.min_tracking_confidence = 0.7  # Only track high-confidence detections
        self.last_tracked_person = None
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "persons_detected": 0,
            "fps": 0,
            "last_detection_time": None
        }
        
        self._setup_routes()
        self._setup_socketio()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page with video stream"""
            return render_template('index.html')
        
        @self.app.route('/stream')
        def video_stream():
            """Video streaming route"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get streaming statistics"""
            return jsonify(self.stats)
        
        @self.app.route('/api/start_stream')
        def start_stream():
            """Start video streaming"""
            if not self.streaming:
                self._start_streaming()
                return jsonify({"status": "started"})
            return jsonify({"status": "already_running"})
        
        @self.app.route('/api/stop_stream')
        def stop_stream():
            """Stop video streaming"""
            if self.streaming:
                self._stop_streaming()
                return jsonify({"status": "stopped"})
            return jsonify({"status": "not_running"})
        
        @self.app.route('/api/detection_status')
        def detection_status():
            """Get current detection status"""
            if self.detector:
                status = self.detector.get_detection_status()
                status.update(self.stats)
                return jsonify(status)
            return jsonify({"error": "detector_not_initialized"})
        
        @self.app.route('/api/camera/center')
        def center_camera_endpoint():
            """Center the camera"""
            try:
                result = center_camera()
                self.socketio.emit('camera_action', {
                    'action': 'center',
                    'result': result,
                    'timestamp': time.time()
                })
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/camera/position')
        def get_camera_position_endpoint():
            """Get current camera position"""
            try:
                result = get_camera_position()
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/camera/move/<direction>')
        def move_camera_endpoint(direction):
            """Move camera in specified direction"""
            try:
                if direction == 'up':
                    result = self.servo_controller.move_up(self.tilt_step_small)
                elif direction == 'down':
                    result = self.servo_controller.move_down(self.tilt_step_small)
                elif direction == 'left':
                    result = self.servo_controller.move_left(self.pan_step_small)
                elif direction == 'right':
                    result = self.servo_controller.move_right(self.pan_step_small)
                else:
                    return jsonify({"error": "invalid_direction"}), 400
                
                self.socketio.emit('camera_action', {
                    'action': f'move_{direction}',
                    'result': result,
                    'timestamp': time.time()
                })
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/tracking/toggle')
        def toggle_tracking():
            """Toggle auto-tracking on/off"""
            self.auto_tracking = not self.auto_tracking
            status = "enabled" if self.auto_tracking else "disabled"
            
            self.socketio.emit('tracking_status', {
                'auto_tracking': self.auto_tracking,
                'message': f'Auto-tracking {status}'
            })
            
            return jsonify({
                "auto_tracking": self.auto_tracking,
                "status": status
            })
        
        @self.app.route('/api/tracking/status')
        def tracking_status():
            """Get current tracking status"""
            return jsonify({
                "auto_tracking": self.auto_tracking,
                "tracking_enabled": self.tracking_enabled,
                "pan_step": self.pan_step_small,
                "tilt_step": self.tilt_step_small,
                "min_confidence": self.min_tracking_confidence,
                "tracking_cooldown": self.tracking_cooldown,
                "deadzone_x": self.center_x_margin,
                "deadzone_y": self.center_y_margin
            })
        
        @self.app.route('/api/tracking/confidence/<float:confidence>')
        def set_tracking_confidence(confidence):
            """Set minimum confidence for tracking"""
            if 0.0 <= confidence <= 1.0:
                self.min_tracking_confidence = confidence
                return jsonify({
                    "min_tracking_confidence": confidence,
                    "status": "updated"
                })
            return jsonify({"error": "confidence must be between 0.0 and 1.0"}), 400
        
        @self.app.route('/api/tracking/cooldown/<float:cooldown>')
        def set_tracking_cooldown(cooldown):
            """Set tracking cooldown period"""
            if cooldown >= 0.0:
                self.tracking_cooldown = cooldown
                return jsonify({
                    "tracking_cooldown": cooldown,
                    "status": "updated"
                })
            return jsonify({"error": "cooldown must be >= 0.0"}), 400
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to video stream")
            emit('status', {'message': 'Connected to EVE Security System'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from video stream")
        
        @self.socketio.on('request_frame')
        def handle_frame_request():
            """Send current frame to client"""
            if self.current_frame is not None:
                with self.frame_lock:
                    frame_data = self._encode_frame(self.current_frame)
                
                emit('frame_data', {
                    'image': frame_data,
                    'detections': self.current_detections,
                    'stats': self.stats
                })
    
    def _start_streaming(self):
        """Start the video streaming"""
        try:
            logger.info("🎥 Starting YOLO video streaming...")
            
            # Initialize YOLO detector
            self.detector = YOLODetector(
                source=self.camera_source,
                confidence_threshold=self.confidence_threshold
            )
            
            # Connect to person notifier
            self.detector.add_detection_callback(self._on_detection_event)
            
            # Start streaming thread
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            logger.info("✅ Video streaming started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.streaming = False
    
    def _stop_streaming(self):
        """Stop the video streaming"""
        logger.info("🛑 Stopping video streaming...")
        self.streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        
        self.detector = None
        logger.info("✅ Video streaming stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            for detection_data in self.detector.stream():
                if not self.streaming:
                    break
                
                # Get frame data
                persons = detection_data.get("persons", [])
                frame_info = detection_data.get("frame", {})
                
                # Update current detections
                self.current_detections = persons
                
                # EVE Auto-tracking logic
                if self.auto_tracking and persons:
                    self._perform_auto_tracking(persons, frame_info)
                
                # Update statistics
                fps_counter += 1
                self.stats["frames_processed"] += 1
                
                if persons:
                    self.stats["persons_detected"] = len(persons)
                    self.stats["last_detection_time"] = time.time()
                else:
                    self.stats["persons_detected"] = 0
                
                # Calculate FPS every second
                if time.time() - fps_start_time >= 1.0:
                    self.stats["fps"] = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Get current frame from detector
                if hasattr(self.detector, 'current_frame') and self.detector.current_frame is not None:
                    with self.detector.frame_lock:
                        frame = self.detector.current_frame.copy()
                    
                    # Draw detection overlays
                    annotated_frame = self._draw_detections(frame, persons)
                    
                    # Store current frame
                    with self.frame_lock:
                        self.current_frame = annotated_frame
                    
                    # Emit frame to connected clients
                    self._emit_frame_update()
                
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.streaming = False
    
    def _draw_detections(self, frame: np.ndarray, persons: list) -> np.ndarray:
        """Draw person detection overlays on frame"""
        annotated_frame = frame.copy()
        
        for i, person in enumerate(persons):
            bbox = person.get("bbox", [0, 0, 0, 0])
            confidence = person.get("conf", 0)
            center = person.get("center", [0, 0])
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for person
            thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Person {i+1}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
        
        # Draw stats overlay
        stats_text = [
            f"FPS: {self.stats['fps']}",
            f"Persons: {len(persons)}",
            f"Total Frames: {self.stats['frames_processed']}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + (i * 25)
            cv2.putText(
                annotated_frame,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_data = base64.b64encode(buffer).decode('utf-8')
            return frame_data
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return ""
    
    def _generate_frames(self):
        """Generate frames for HTTP streaming"""
        while True:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame = self.current_frame.copy()
                
                # Encode frame as JPEG
                try:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_data = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except Exception as e:
                    logger.error(f"Error generating frame: {e}")
            
            time.sleep(0.033)  # ~30 FPS
    
    def _emit_frame_update(self):
        """Emit frame update to SocketIO clients"""
        try:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame_data = self._encode_frame(self.current_frame)
                
                self.socketio.emit('frame_update', {
                    'image': frame_data,
                    'detections': self.current_detections,
                    'stats': self.stats
                })
        except Exception as e:
            logger.error(f"Error emitting frame update: {e}")
    
    def _perform_auto_tracking(self, persons: list, frame_info: dict):
        """Perform automatic camera tracking based on detected persons"""
        try:
            current_time = time.time()
            
            # Check if tracking is enabled
            if not self.auto_tracking:
                return
            
            # Check cooldown to prevent jittery movement
            if current_time - self.last_tracking_time < self.tracking_cooldown:
                return
            
            # No persons detected
            if not persons:
                logger.debug("No persons detected for tracking")
                return
            
            # Filter persons by confidence threshold
            confident_persons = [p for p in persons if p.get("conf", 0) >= self.min_tracking_confidence]
            
            if not confident_persons:
                logger.debug(f"No persons above confidence threshold {self.min_tracking_confidence}")
                return
            
            # Get the most confident person for tracking
            best_person = max(confident_persons, key=lambda p: p.get("conf", 0))
            bbox = best_person.get("bbox", [0, 0, 0, 0])
            confidence = best_person.get("conf", 0)
            
            logger.info(f"Tracking person with {confidence:.1%} confidence at bbox: {bbox}")
            
            frame_w = frame_info.get("w", 640)
            frame_h = frame_info.get("h", 480)
            
            # Apply the nudge-to-center logic
            moved = self._nudge_to_center(bbox, frame_w, frame_h, best_person)
            
            # Update tracking time only if we actually moved
            if moved:
                self.last_tracking_time = current_time
                self.last_tracked_person = best_person
                
                # Emit tracking action to web clients
                self.socketio.emit('auto_tracking', {
                    'person': best_person,
                    'frame_size': {'w': frame_w, 'h': frame_h},
                    'timestamp': current_time,
                    'moved': True
                })
            
        except Exception as e:
            logger.error(f"Error in auto-tracking: {e}")
    
    def _nudge_to_center(self, bbox: list, frame_w: int, frame_h: int, person_data: dict = None) -> bool:
        """Nudge camera to center the person (improved logic)"""
        try:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # Frame center
            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
            
            # Calculate offsets from center
            offset_x = cx - center_x
            offset_y = cy - center_y
            
            # Define movement thresholds (deadzone)
            x_threshold = frame_w * self.center_x_margin
            y_threshold = frame_h * self.center_y_margin
            
            moved = False
            actions = []

            # Horizontal movement
            if abs(offset_x) > x_threshold:
                if offset_x > 0:  # Person is to the right, move camera right
                    result = self.servo_controller.move_right(self.pan_step_small)
                    actions.append("move_right")
                    moved = True
                    logger.info(f"Moving camera RIGHT - person at x={cx:.0f}, center={center_x:.0f}, offset={offset_x:.0f}")
                else:  # Person is to the left, move camera left
                    result = self.servo_controller.move_left(self.pan_step_small)
                    actions.append("move_left")
                    moved = True
                    logger.info(f"Moving camera LEFT - person at x={cx:.0f}, center={center_x:.0f}, offset={offset_x:.0f}")
            else:
                logger.debug(f"Horizontal CENTERED - person at x={cx:.0f}, center={center_x:.0f}, offset={offset_x:.0f}")

            # Vertical movement (y=0 is top)
            if abs(offset_y) > y_threshold:
                if offset_y > 0:  # Person is below center, move camera down
                    result = self.servo_controller.move_down(self.tilt_step_small)
                    actions.append("move_down")
                    moved = True
                    logger.info(f"Moving camera DOWN - person at y={cy:.0f}, center={center_y:.0f}, offset={offset_y:.0f}")
                else:  # Person is above center, move camera up
                    result = self.servo_controller.move_up(self.tilt_step_small)
                    actions.append("move_up")
                    moved = True
                    logger.info(f"Moving camera UP - person at y={cy:.0f}, center={center_y:.0f}, offset={offset_y:.0f}")
            else:
                logger.debug(f"Vertical CENTERED - person at y={cy:.0f}, center={center_y:.0f}, offset={offset_y:.0f}")
            
            if moved:
                logger.info(f"🎯 Camera tracking: {', '.join(actions)} - Person confidence: {person_data.get('conf', 0):.1%}")
            else:
                logger.debug(f"👁️  Person already centered - no movement needed")
            
            return moved
            
        except Exception as e:
            logger.error(f"Error in nudge_to_center: {e}")
            return False
    
    def _on_detection_event(self, event_data: Dict[str, Any]):
        """Handle detection events from YOLO detector"""
        try:
            # Forward to person notifier (which handles EVE agent notifications)
            person_notifier.on_person_detected(event_data)
            
            # Handle security events
            event_type = event_data.get('event')
            person_count = event_data.get('person_count', 0)
            
            if event_type == 'person_entered':
                # Set alert mode when person enters
                try:
                    set_alert_mode(True)
                    logger.info("🚨 EVE Alert Mode activated - person detected")
                except Exception as e:
                    logger.error(f"Failed to set alert mode: {e}")
            
            elif event_type == 'person_left':
                # Return to normal mode when all persons leave
                try:
                    set_alert_mode(False)
                    center_camera()
                    logger.info("📹 EVE returning to normal surveillance mode")
                except Exception as e:
                    logger.error(f"Failed to reset to normal mode: {e}")
            
            # Emit detection event to web clients
            self.socketio.emit('detection_event', {
                'event': event_type,
                'person_count': person_count,
                'timestamp': event_data.get('timestamp', time.time()),
                'persons': event_data.get('persons', []),
                'eve_response': True  # Indicate EVE system responded
            })
            
        except Exception as e:
            logger.error(f"Error handling detection event: {e}")
    
    def run(self, debug=False):
        """Run the streaming server"""
        logger.info(f"🚀 Starting EVE Video Streaming Server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

def create_streaming_server(camera_source="/dev/video0", port=5000):
    """Create and return a video streaming server instance"""
    return YOLOVideoStreamer(camera_source=camera_source, port=port)