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
        
        # Streaming control
        self.streaming = False
        self.stream_thread = None
        
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
    
    def _on_detection_event(self, event_data: Dict[str, Any]):
        """Handle detection events from YOLO detector"""
        try:
            # Forward to person notifier
            person_notifier.on_person_detected(event_data)
            
            # Emit detection event to web clients
            self.socketio.emit('detection_event', {
                'event': event_data.get('event'),
                'person_count': event_data.get('person_count', 0),
                'timestamp': event_data.get('timestamp', time.time()),
                'persons': event_data.get('persons', [])
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