"""
AI description service for anomaly detection events
Generates detailed descriptions of security events using AI vision models
"""
import os
import cv2
import base64
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import google.generativeai as genai
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIDescription:
    """Represents an AI-generated description"""
    timestamp: float
    event_id: str
    description: str
    confidence: float
    analysis_type: str  # 'image' or 'video'
    prompt_used: str
    model_used: str

class AIDescriptionService:
    """
    Service for generating AI descriptions of anomaly events
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the AI description service
        
        Args:
            api_key: Google AI API key (will try to get from environment if not provided)
            model_name: Name of the AI model to use
        """
        self.model_name = model_name
        self.model = None
        
        # Configure Google AI
        try:
            api_key = api_key or os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"AI description service initialized with model: {model_name}")
            else:
                logger.warning("No Google AI API key found. AI descriptions will be disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI: {e}")
            self.model = None
    
    async def generate_description(self, 
                                 capture_event,
                                 anomaly_event,
                                 additional_context: Optional[str] = None) -> Optional[AIDescription]:
        """
        Generate an AI description for a capture event
        
        Args:
            capture_event: CaptureEvent object
            anomaly_event: AnomalyEvent object  
            additional_context: Additional context to include in the prompt
            
        Returns:
            AIDescription object or None if generation failed
        """
        if not self.model:
            logger.warning("AI model not available for description generation")
            return None
        
        try:
            if capture_event.media_type == 'screenshot':
                return await self._generate_image_description(capture_event, anomaly_event, additional_context)
            elif capture_event.media_type == 'video':
                return await self._generate_video_description(capture_event, anomaly_event, additional_context)
            else:
                logger.error(f"Unsupported media type: {capture_event.media_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating AI description: {e}")
            return None
    
    async def _generate_image_description(self, 
                                        capture_event,
                                        anomaly_event, 
                                        additional_context: Optional[str] = None) -> Optional[AIDescription]:
        """Generate description for an image/screenshot"""
        try:
            # Read and encode image
            image_data = self._encode_image(capture_event.file_path)
            if not image_data:
                return None
            
            # Create prompt
            prompt = self._create_image_prompt(anomaly_event, additional_context)
            
            # Generate description
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, {"mime_type": "image/jpeg", "data": image_data}]
            )
            
            if response and response.text:
                return AIDescription(
                    timestamp=capture_event.timestamp,
                    event_id=capture_event.event_id,
                    description=response.text.strip(),
                    confidence=0.8,  # Default confidence
                    analysis_type='image',
                    prompt_used=prompt,
                    model_used=self.model_name
                )
            else:
                logger.error("Empty response from AI model")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return None
    
    async def _generate_video_description(self, 
                                        capture_event,
                                        anomaly_event,
                                        additional_context: Optional[str] = None) -> Optional[AIDescription]:
        """Generate description for a video - extracts key frames and analyzes them"""
        try:
            # Extract key frames from video
            key_frames = self._extract_key_frames(capture_event.file_path)
            if not key_frames:
                logger.error("No key frames extracted from video")
                return None
            
            # Analyze key frames
            frame_descriptions = []
            for i, frame_data in enumerate(key_frames):
                try:
                    prompt = self._create_video_frame_prompt(anomaly_event, i, len(key_frames), additional_context)
                    
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        [prompt, {"mime_type": "image/jpeg", "data": frame_data}]
                    )
                    
                    if response and response.text:
                        frame_descriptions.append(response.text.strip())
                except Exception as e:
                    logger.error(f"Error analyzing frame {i}: {e}")
                    continue
            
            if frame_descriptions:
                # Combine frame descriptions into a coherent video description
                combined_description = self._combine_frame_descriptions(
                    frame_descriptions, anomaly_event, capture_event
                )
                
                return AIDescription(
                    timestamp=capture_event.timestamp,
                    event_id=capture_event.event_id,
                    description=combined_description,
                    confidence=0.7,  # Slightly lower confidence for video analysis
                    analysis_type='video',
                    prompt_used="Video frame analysis",
                    model_used=self.model_name
                )
            else:
                logger.error("No frame descriptions generated")
                return None
                
        except Exception as e:
            logger.error(f"Error generating video description: {e}")
            return None
    
    def _encode_image(self, image_path: str) -> Optional[bytes]:
        """Encode image to base64 bytes"""
        try:
            with open(image_path, 'rb') as image_file:
                return image_file.read()
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def _extract_key_frames(self, video_path: str, max_frames: int = 5) -> List[bytes]:
        """Extract key frames from video for analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error("Video has no frames")
                return []
            
            # Calculate frame indices to extract
            frame_indices = []
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames // max_frames
                frame_indices = [i * step for i in range(max_frames)]
            
            key_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Encode frame as JPEG
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        key_frames.append(buffer.tobytes())
            
            cap.release()
            return key_frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames from {video_path}: {e}")
            return []
    
    def _create_image_prompt(self, anomaly_event, additional_context: Optional[str] = None) -> str:
        """Create prompt for image analysis"""
        timestamp_str = datetime.fromtimestamp(anomaly_event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""Analyze this security camera image captured during a {anomaly_event.anomaly_type} anomaly detection event.

Event Details:
- Anomaly Type: {anomaly_event.anomaly_type}
- Timestamp: {timestamp_str}
- Severity: {anomaly_event.severity}
- Person Count: {anomaly_event.person_count}
- Confidence: {anomaly_event.confidence:.1%}

Please provide a detailed description including:
1. What you observe in the image
2. Number and appearance of people visible
3. Their activities and behaviors
4. Any suspicious or unusual activities
5. Environmental context (location, lighting, etc.)
6. Potential security concerns or risks

"""
        
        if additional_context:
            prompt += f"Additional Context: {additional_context}\n\n"
        
        prompt += "Provide a comprehensive but concise security-focused analysis suitable for incident reporting."
        
        return prompt
    
    def _create_video_frame_prompt(self, anomaly_event, frame_index: int, total_frames: int, additional_context: Optional[str] = None) -> str:
        """Create prompt for video frame analysis"""
        prompt = f"""Analyze this frame ({frame_index + 1} of {total_frames}) from a security video showing a {anomaly_event.anomaly_type} event.

Focus on:
1. People and their activities in this frame
2. Any movement or behavior patterns
3. Changes from what you might expect in previous frames
4. Security-relevant observations

Provide a brief but informative description of this specific frame."""
        
        return prompt
    
    def _combine_frame_descriptions(self, frame_descriptions: List[str], anomaly_event, capture_event) -> str:
        """Combine individual frame descriptions into a coherent video description"""
        timestamp_str = datetime.fromtimestamp(anomaly_event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        combined = f"""Security Video Analysis - {anomaly_event.anomaly_type.replace('_', ' ').title()}
Recorded: {timestamp_str}
Duration: {capture_event.duration:.1f} seconds
Severity: {anomaly_event.severity.upper()}

Video Summary:
"""
        
        for i, description in enumerate(frame_descriptions):
            combined += f"\nFrame {i+1}: {description}"
        
        # Add overall assessment
        combined += f"\n\nOverall Assessment: This {capture_event.duration:.1f}-second video clip shows a {anomaly_event.anomaly_type} event involving {anomaly_event.person_count} person(s). "
        
        if anomaly_event.severity in ['high', 'critical']:
            combined += "This event requires immediate attention and further investigation."
        elif anomaly_event.severity == 'medium':
            combined += "This event should be reviewed and monitored."
        else:
            combined += "This event has been logged for routine review."
        
        return combined
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get AI service status"""
        return {
            'model_available': self.model is not None,
            'model_name': self.model_name,
            'api_configured': os.getenv('GOOGLE_AI_API_KEY') is not None or os.getenv('GEMINI_API_KEY') is not None
        }

# Global instance for easy access
ai_description_service = AIDescriptionService()