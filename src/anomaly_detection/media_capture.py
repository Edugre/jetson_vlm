"""
Media capture service for anomaly detection events
Captures screenshots and video clips when anomalies are detected
"""
import os
import cv2
import time
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CaptureEvent:
    """Represents a media capture event"""
    timestamp: float
    event_id: str
    anomaly_type: str
    media_type: str  # 'screenshot' or 'video'
    file_path: str
    file_size: int
    duration: Optional[float] = None  # For videos
    frame_count: Optional[int] = None  # For videos

class MediaCaptureService:
    """
    Service for capturing screenshots and videos during anomaly events
    """
    
    def __init__(self, 
                 capture_dir: str = "/tmp/anomaly_captures",
                 video_duration: float = 10.0,
                 fps: float = 15.0,
                 enable_video: bool = True,
                 enable_screenshots: bool = True):
        """
        Initialize the media capture service
        
        Args:
            capture_dir: Directory to store captured media
            video_duration: Duration of video clips in seconds
            fps: Frames per second for video recording
            enable_video: Whether to capture video clips
            enable_screenshots: Whether to capture screenshots
        """
        self.capture_dir = capture_dir
        self.video_duration = video_duration
        self.fps = fps
        self.enable_video = enable_video
        self.enable_screenshots = enable_screenshots
        
        # Create capture directory
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Video recording state
        self.recording_lock = threading.Lock()
        self.active_recordings = {}
        
        # Frame buffer for video recording
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = int(fps * video_duration * 2)  # 2x duration for safety
        
        # Capture callbacks
        self.capture_callbacks = []
        
        logger.info(f"Media capture service initialized - Dir: {capture_dir}")
    
    def add_capture_callback(self, callback):
        """Add callback for capture events"""
        self.capture_callbacks.append(callback)
    
    def update_frame_buffer(self, frame: np.ndarray):
        """
        Update the frame buffer with the latest frame
        Call this continuously to maintain a rolling buffer for video capture
        """
        with self.buffer_lock:
            self.frame_buffer.append({
                'frame': frame.copy(),
                'timestamp': time.time()
            })
            
            # Maintain buffer size
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
    
    def capture_anomaly_media(self, 
                            anomaly_event,
                            current_frame: Optional[np.ndarray] = None) -> List[CaptureEvent]:
        """
        Capture media for an anomaly event
        
        Args:
            anomaly_event: AnomalyEvent object
            current_frame: Current frame from camera
            
        Returns:
            List of CaptureEvent objects for captured media
        """
        captured_events = []
        
        try:
            # Generate unique event ID
            event_id = f"{anomaly_event.anomaly_type}_{int(anomaly_event.timestamp)}"
            timestamp_str = datetime.fromtimestamp(anomaly_event.timestamp).strftime("%Y%m%d_%H%M%S")
            
            # Capture screenshot
            if self.enable_screenshots and current_frame is not None:
                screenshot_event = self._capture_screenshot(
                    event_id, timestamp_str, anomaly_event.anomaly_type, current_frame
                )
                if screenshot_event:
                    captured_events.append(screenshot_event)
            
            # Capture video clip
            if self.enable_video:
                video_event = self._capture_video_clip(
                    event_id, timestamp_str, anomaly_event.anomaly_type
                )
                if video_event:
                    captured_events.append(video_event)
            
            # Trigger callbacks
            for event in captured_events:
                self._trigger_capture_callbacks(event, anomaly_event)
            
            logger.info(f"Captured {len(captured_events)} media files for anomaly: {anomaly_event.anomaly_type}")
            
        except Exception as e:
            logger.error(f"Error capturing media for anomaly: {e}")
        
        return captured_events
    
    def _capture_screenshot(self, 
                          event_id: str, 
                          timestamp_str: str, 
                          anomaly_type: str, 
                          frame: np.ndarray) -> Optional[CaptureEvent]:
        """Capture a screenshot"""
        try:
            filename = f"screenshot_{anomaly_type}_{timestamp_str}.jpg"
            file_path = os.path.join(self.capture_dir, filename)
            
            # Save screenshot
            success = cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                
                return CaptureEvent(
                    timestamp=time.time(),
                    event_id=event_id,
                    anomaly_type=anomaly_type,
                    media_type='screenshot',
                    file_path=file_path,
                    file_size=file_size
                )
            else:
                logger.error(f"Failed to save screenshot: {file_path}")
                
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
        
        return None
    
    def _capture_video_clip(self, 
                          event_id: str, 
                          timestamp_str: str, 
                          anomaly_type: str) -> Optional[CaptureEvent]:
        """Capture a video clip from the frame buffer"""
        try:
            with self.recording_lock:
                # Check if already recording this type
                recording_key = f"{anomaly_type}_{int(time.time() // 30)}"  # Group recordings by 30s intervals
                if recording_key in self.active_recordings:
                    logger.debug(f"Already recording {anomaly_type}, skipping duplicate")
                    return None
                
                # Mark as recording
                self.active_recordings[recording_key] = True
            
            try:
                filename = f"video_{anomaly_type}_{timestamp_str}.mp4"
                file_path = os.path.join(self.capture_dir, filename)
                
                # Get frames from buffer
                with self.buffer_lock:
                    if not self.frame_buffer:
                        logger.warning("No frames in buffer for video capture")
                        return None
                    
                    # Calculate how many frames we need
                    target_frames = int(self.fps * self.video_duration)
                    available_frames = len(self.frame_buffer)
                    
                    if available_frames < target_frames // 2:  # At least half the desired frames
                        logger.warning(f"Insufficient frames for video: {available_frames} < {target_frames//2}")
                        return None
                    
                    # Take the most recent frames
                    frames_to_use = self.frame_buffer[-target_frames:] if available_frames >= target_frames else self.frame_buffer[:]
                
                # Create video
                if frames_to_use:
                    first_frame = frames_to_use[0]['frame']
                    height, width, _ = first_frame.shape
                    
                    # Define codec and create VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(file_path, fourcc, self.fps, (width, height))
                    
                    frame_count = 0
                    for frame_data in frames_to_use:
                        out.write(frame_data['frame'])
                        frame_count += 1
                    
                    out.release()
                    
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        actual_duration = frame_count / self.fps
                        
                        return CaptureEvent(
                            timestamp=time.time(),
                            event_id=event_id,
                            anomaly_type=anomaly_type,
                            media_type='video',
                            file_path=file_path,
                            file_size=file_size,
                            duration=actual_duration,
                            frame_count=frame_count
                        )
                    else:
                        logger.error(f"Video file not created: {file_path}")
                
            finally:
                # Remove from active recordings after a delay
                def cleanup_recording():
                    time.sleep(30)  # Wait 30 seconds before allowing another recording
                    with self.recording_lock:
                        self.active_recordings.pop(recording_key, None)
                
                threading.Thread(target=cleanup_recording, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error capturing video clip: {e}")
        
        return None
    
    def _trigger_capture_callbacks(self, capture_event: CaptureEvent, anomaly_event):
        """Trigger callbacks for capture events"""
        for callback in self.capture_callbacks:
            try:
                callback(capture_event, anomaly_event)
            except Exception as e:
                logger.error(f"Error in capture callback: {e}")
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old capture files"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            removed_count = 0
            for filename in os.listdir(self.capture_dir):
                file_path = os.path.join(self.capture_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                            logger.debug(f"Removed old capture file: {filename}")
                        except Exception as e:
                            logger.error(f"Error removing file {filename}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old capture files")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture service statistics"""
        try:
            # Count files by type
            screenshot_count = 0
            video_count = 0
            total_size = 0
            
            if os.path.exists(self.capture_dir):
                for filename in os.listdir(self.capture_dir):
                    file_path = os.path.join(self.capture_dir, filename)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                        if filename.startswith('screenshot_'):
                            screenshot_count += 1
                        elif filename.startswith('video_'):
                            video_count += 1
            
            return {
                'capture_dir': self.capture_dir,
                'screenshot_count': screenshot_count,
                'video_count': video_count,
                'total_size_mb': total_size / (1024 * 1024),
                'buffer_size': len(self.frame_buffer),
                'active_recordings': len(self.active_recordings),
                'enable_video': self.enable_video,
                'enable_screenshots': self.enable_screenshots,
                'video_duration': self.video_duration,
                'fps': self.fps
            }
            
        except Exception as e:
            logger.error(f"Error getting capture stats: {e}")
            return {}

# Global instance for easy access
media_capture_service = MediaCaptureService()