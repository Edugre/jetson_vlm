"""
Anomaly detection system for security monitoring
Detects unusual patterns in person detection and triggers alerts
"""
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyEvent:
    """Represents an anomaly detection event"""
    timestamp: float
    anomaly_type: str
    confidence: float
    person_count: int
    persons: List[Dict[str, Any]]
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'

class AnomalyDetector:
    """
    Detects anomalies in person detection patterns
    """
    
    def __init__(self):
        """Initialize the anomaly detector"""
        self.detection_history = []
        self.max_history_size = 1000
        self.anomaly_callbacks = []
        
        # Anomaly detection parameters
        self.normal_person_count_threshold = 3  # More than 3 people is unusual
        self.rapid_entry_threshold = 5  # 5 people entering within time window
        self.rapid_entry_window = 30  # seconds
        self.loitering_threshold = 300  # 5 minutes of continuous presence
        self.suspicious_movement_threshold = 10  # Rapid position changes
        
        # Tracking variables
        self.continuous_presence_start = None
        self.last_person_positions = {}
        self.rapid_entries = []
        
    def add_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]):
        """Add callback function for anomaly events"""
        self.anomaly_callbacks.append(callback)
    
    def process_detection_event(self, event_data: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """
        Process a detection event and check for anomalies
        
        Args:
            event_data: Detection event from YOLO detector
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        try:
            # Add to history
            self._add_to_history(event_data)
            
            # Check for different types of anomalies
            anomaly = None
            
            # Check for multiple person anomaly
            if not anomaly:
                anomaly = self._check_multiple_persons(event_data)
            
            # Check for rapid entry anomaly
            if not anomaly:
                anomaly = self._check_rapid_entries(event_data)
            
            # Check for loitering anomaly
            if not anomaly:
                anomaly = self._check_loitering(event_data)
            
            # Check for suspicious movement patterns
            if not anomaly:
                anomaly = self._check_suspicious_movement(event_data)
            
            # Trigger callbacks if anomaly detected
            if anomaly:
                self._trigger_anomaly_callbacks(anomaly)
                logger.warning(f"🚨 ANOMALY DETECTED: {anomaly.anomaly_type} - {anomaly.description}")
            
            return anomaly
            
        except Exception as e:
            logger.error(f"Error processing detection event for anomalies: {e}")
            return None
    
    def _check_multiple_persons(self, event_data: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Check for unusual number of people"""
        person_count = event_data.get('person_count', 0)
        
        if person_count > self.normal_person_count_threshold:
            severity = 'high' if person_count > 5 else 'medium'
            
            return AnomalyEvent(
                timestamp=event_data.get('timestamp', time.time()),
                anomaly_type='multiple_persons',
                confidence=0.9,
                person_count=person_count,
                persons=event_data.get('persons', []),
                description=f"Unusual number of people detected: {person_count} individuals",
                severity=severity
            )
        
        return None
    
    def _check_rapid_entries(self, event_data: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Check for rapid succession of people entering"""
        event_type = event_data.get('event')
        current_time = event_data.get('timestamp', time.time())
        
        if event_type == 'person_entered':
            # Add to rapid entries list
            self.rapid_entries.append(current_time)
            
            # Clean old entries outside time window
            cutoff_time = current_time - self.rapid_entry_window
            self.rapid_entries = [t for t in self.rapid_entries if t > cutoff_time]
            
            # Check if threshold exceeded
            if len(self.rapid_entries) >= self.rapid_entry_threshold:
                return AnomalyEvent(
                    timestamp=current_time,
                    anomaly_type='rapid_entries',
                    confidence=0.8,
                    person_count=event_data.get('person_count', 0),
                    persons=event_data.get('persons', []),
                    description=f"Rapid succession of entries: {len(self.rapid_entries)} people in {self.rapid_entry_window} seconds",
                    severity='critical'
                )
        
        return None
    
    def _check_loitering(self, event_data: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Check for prolonged presence (loitering)"""
        event_type = event_data.get('event')
        current_time = event_data.get('timestamp', time.time())
        person_count = event_data.get('person_count', 0)
        
        if event_type == 'person_entered' and person_count > 0:
            if self.continuous_presence_start is None:
                self.continuous_presence_start = current_time
        elif event_type == 'person_left':
            self.continuous_presence_start = None
        
        # Check for loitering
        if (self.continuous_presence_start and 
            current_time - self.continuous_presence_start > self.loitering_threshold):
            
            duration = current_time - self.continuous_presence_start
            
            return AnomalyEvent(
                timestamp=current_time,
                anomaly_type='loitering',
                confidence=0.7,
                person_count=person_count,
                persons=event_data.get('persons', []),
                description=f"Prolonged presence detected: {duration/60:.1f} minutes",
                severity='medium'
            )
        
        return None
    
    def _check_suspicious_movement(self, event_data: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Check for suspicious movement patterns"""
        persons = event_data.get('persons', [])
        current_time = event_data.get('timestamp', time.time())
        
        for person in persons:
            center = person.get('center', [0, 0])
            person_id = f"person_{hash(str(center))}"  # Simple ID based on position
            
            if person_id in self.last_person_positions:
                last_pos, last_time = self.last_person_positions[person_id]
                
                # Calculate movement speed
                distance = ((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)**0.5
                time_diff = current_time - last_time
                
                if time_diff > 0:
                    speed = distance / time_diff
                    
                    # Check for unusually rapid movement
                    if speed > self.suspicious_movement_threshold:
                        return AnomalyEvent(
                            timestamp=current_time,
                            anomaly_type='suspicious_movement',
                            confidence=0.6,
                            person_count=len(persons),
                            persons=persons,
                            description=f"Rapid movement detected: speed {speed:.1f} units/second",
                            severity='low'
                        )
            
            # Update position tracking
            self.last_person_positions[person_id] = (center, current_time)
        
        return None
    
    def _add_to_history(self, event_data: Dict[str, Any]):
        """Add event to detection history"""
        self.detection_history.append(event_data)
        
        # Maintain history size limit
        if len(self.detection_history) > self.max_history_size:
            self.detection_history = self.detection_history[-self.max_history_size:]
    
    def _trigger_anomaly_callbacks(self, anomaly_event: AnomalyEvent):
        """Trigger all registered anomaly callbacks"""
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly_event)
            except Exception as e:
                logger.error(f"Error in anomaly callback: {e}")
    
    def get_recent_anomalies(self, limit: int = 10) -> List[AnomalyEvent]:
        """Get recent anomaly events"""
        # For now, we don't store anomaly history separately
        # This could be enhanced to maintain a separate anomaly history
        return []
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current anomaly detection configuration"""
        return {
            'normal_person_count_threshold': self.normal_person_count_threshold,
            'rapid_entry_threshold': self.rapid_entry_threshold,
            'rapid_entry_window': self.rapid_entry_window,
            'loitering_threshold': self.loitering_threshold,
            'suspicious_movement_threshold': self.suspicious_movement_threshold
        }
    
    def update_configuration(self, config: Dict[str, Any]):
        """Update anomaly detection configuration"""
        if 'normal_person_count_threshold' in config:
            self.normal_person_count_threshold = config['normal_person_count_threshold']
        if 'rapid_entry_threshold' in config:
            self.rapid_entry_threshold = config['rapid_entry_threshold']
        if 'rapid_entry_window' in config:
            self.rapid_entry_window = config['rapid_entry_window']
        if 'loitering_threshold' in config:
            self.loitering_threshold = config['loitering_threshold']
        if 'suspicious_movement_threshold' in config:
            self.suspicious_movement_threshold = config['suspicious_movement_threshold']
        
        logger.info("Anomaly detection configuration updated")

# Global instance for easy access
anomaly_detector = AnomalyDetector()