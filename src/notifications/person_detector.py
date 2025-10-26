"""
Person detection notification system for EVE security camera
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from adk_backend.agent import root_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetectionNotifier:
    """Handles person detection events and notifies the ADK agent"""
    
    def __init__(self, agent=None):
        """
        Initialize the person detection notifier
        
        Args:
            agent: ADK agent instance (defaults to root_agent)
        """
        self.agent = agent or root_agent
        self.last_notification_time = 0
        self.notification_cooldown = 2.0  # seconds
        
        # Event tracking
        self.event_history = []
        self.max_history = 50
        
    def on_person_detected(self, event_data: Dict[str, Any]):
        """
        Callback for person detection events from YOLO detector
        
        Args:
            event_data: Detection event data from YOLODetector
        """
        try:
            event_type = event_data.get("event")
            person_count = event_data.get("person_count", 0)
            persons = event_data.get("persons", [])
            timestamp = event_data.get("timestamp", time.time())
            
            # Add to history
            self._add_to_history(event_data)
            
            # Check cooldown
            if timestamp - self.last_notification_time < self.notification_cooldown:
                logger.debug(f"Skipping notification due to cooldown: {event_type}")
                return
            
            # Process different event types
            if event_type == "person_entered":
                self._handle_person_entered(person_count, persons, timestamp)
            elif event_type == "person_left":
                self._handle_person_left(timestamp)
            elif event_type == "person_count_changed":
                self._handle_person_count_changed(person_count, persons, timestamp)
            
            self.last_notification_time = timestamp
            
        except Exception as e:
            logger.error(f"Error processing person detection event: {e}")
    
    def _handle_person_entered(self, person_count: int, persons: list, timestamp: float):
        """Handle person entering the frame"""
        logger.info(f"🚨 SECURITY ALERT: {person_count} person(s) detected in frame!")
        
        # Get the most confident detection for positioning
        if persons:
            best_person = max(persons, key=lambda p: p.get("conf", 0))
            bbox = best_person.get("bbox", [0, 0, 0, 0])
            center = best_person.get("center", [0, 0])
            confidence = best_person.get("conf", 0)
            
            # Prepare message for ADK agent
            message = (
                f"SECURITY ALERT: {person_count} person(s) detected! "
                f"Primary target at position ({int(center[0])}, {int(center[1])}) "
                f"with {confidence:.1%} confidence. "
                f"Please track this person and adjust camera positioning."
            )
        else:
            message = f"SECURITY ALERT: {person_count} person(s) detected! Please scan the area."
        
        # Send to ADK agent asynchronously
        asyncio.create_task(self._notify_agent(message))
    
    def _handle_person_left(self, timestamp: float):
        """Handle all persons leaving the frame"""
        logger.info("ℹ️  All persons have left the monitoring area")
        
        message = (
            "All persons have left the monitoring area. "
            "Returning to default surveillance mode. "
            "Continue monitoring for new activity."
        )
        
        # Send to ADK agent asynchronously
        asyncio.create_task(self._notify_agent(message))
    
    def _handle_person_count_changed(self, person_count: int, persons: list, timestamp: float):
        """Handle change in person count"""
        logger.info(f"👥 Person count changed to {person_count}")
        
        message = (
            f"Person count has changed to {person_count}. "
            f"Please adjust camera positioning to monitor all individuals. "
            f"Maintain visual contact with all targets."
        )
        
        # Send to ADK agent asynchronously
        asyncio.create_task(self._notify_agent(message))
    
    async def _notify_agent(self, message: str):
        """Send notification to ADK agent"""
        try:
            logger.info(f"Notifying ADK Agent: {message}")
            
            # Send message to the agent - run_async returns an async generator
            response_generator = self.agent.run_async(message)
            
            # Collect all responses from the generator
            responses = []
            async for response in response_generator:
                responses.append(response)
            
            if responses:
                # Get the final response (usually the last one)
                final_response = responses[-1] if responses else None
                logger.info(f"ADK Agent response: {final_response}")
            else:
                logger.warning("No response from ADK Agent")
                
        except Exception as e:
            logger.error(f"Failed to notify ADK Agent: {e}")
    
    def _add_to_history(self, event_data: Dict[str, Any]):
        """Add event to history with size limit"""
        self.event_history.append(event_data)
        
        # Maintain history size limit
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    def get_recent_events(self, limit: int = 10) -> list:
        """Get recent detection events"""
        return self.event_history[-limit:] if self.event_history else []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current notifier status"""
        return {
            "last_notification_time": self.last_notification_time,
            "notification_cooldown": self.notification_cooldown,
            "event_count": len(self.event_history),
            "agent_connected": self.agent is not None
        }
    
    def set_cooldown(self, cooldown_seconds: float):
        """Set notification cooldown period"""
        self.notification_cooldown = max(0.0, cooldown_seconds)
        logger.info(f"Notification cooldown set to {self.notification_cooldown} seconds")

# Global instance for easy access
person_notifier = PersonDetectionNotifier()