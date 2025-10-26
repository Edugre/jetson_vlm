#!/usr/bin/env python3
"""
EVE Security System - Live Video Streaming Application
Launch script for the YOLO video streaming frontend
"""
import sys
import os
import logging
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming.video_server import YOLOVideoStreamer

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="EVE Security System - Live Video Stream")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device path (default: /dev/video0)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 EVE Security System - Live Video Stream")
    print("=" * 50)
    print(f"📹 Camera: {args.camera}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"🎯 Confidence: {args.confidence}")
    print("=" * 50)
    
    try:
        # Create and run the streaming server
        server = YOLOVideoStreamer(
            camera_source=args.camera,
            confidence_threshold=args.confidence,
            host=args.host,
            port=args.port
        )
        
        print("🎥 Starting video streaming server...")
        print(f"📱 Open your browser to: http://localhost:{args.port}")
        print("⏹️  Press Ctrl+C to stop")
        
        server.run(debug=args.debug)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down EVE Security System...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()