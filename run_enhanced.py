import os
import sys
import webbrowser
import time
from pathlib import Path
import uvicorn
import argparse


def setup_directories():
    """Ensure all required directories exist"""
    dirs = [
        Path("../videos"),
        Path("../faces"),
        Path("./uploads"),
        Path("../output"),
        Path("./temp"),
        Path("./static"),
        Path("./templates")
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✅ All directories created/verified")


def check_sample_videos():
    """Check if there are videos in the ../videos directory"""
    videos_dir = Path("../videos")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []

    for ext in video_extensions:
        videos.extend(videos_dir.glob(f"*{ext}"))

    if not videos:
        print("\n⚠️  WARNING: No videos found in ../videos directory")
        print("   Please add some video files to ../videos before starting")
        print("   Supported formats: MP4, AVI, MOV, MKV\n")
        return False
    else:
        print(f"✅ Found {len(videos)} video(s) in ../videos")
        for video in videos[:5]:  # Show first 5
            print(f"   - {video.name}")
        if len(videos) > 5:
            print(f"   ... and {len(videos) - 5} more")
    return True


def clear_old_faces():
    """Clear any existing faces from previous sessions"""
    faces_dir = Path("../faces")
    if faces_dir.exists():
        for file in faces_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print("✅ Cleared old face images for privacy")


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 60)
    print("   🎭 ROPE DEEPFAKE STUDIO - ENHANCED VERSION")
    print("=" * 60)
    print("   WebCam Support | High Quality Output | Modern UI")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run Enhanced Rope Deepfake System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--no-browser', action='store_true', help="Don't open browser automatically")
    parser.add_argument('--keep-faces', action='store_true', help="Don't clear existing faces on startup")
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Setup
    print("🔧 Setting up environment...\n")
    setup_directories()

    # Check for videos
    has_videos = check_sample_videos()

    # Clear old faces unless specified
    if not args.keep_faces:
        clear_old_faces()

    print("\n" + "=" * 60)
    print("🚀 Starting Enhanced Rope API Server")
    print("=" * 60)
    print(f"   Address: http://{args.host}:{args.port}")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # Open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            webbrowser.open(f'http://localhost:{args.port}')

        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

    print("\n📌 CONTROLS:")
    print("   - Press Ctrl+C to stop the server")
    print("   - Webcam recording captures for 5 seconds")
    print("   - Faces are auto-cleared for privacy")
    print("\n🎥 QUALITY TIPS:")
    print("   - Use CRF 10-14 for best quality")
    print("   - Use 'slow' or 'veryslow' preset for better results")
    print("   - More threads = faster processing")

    if not has_videos:
        print("\n⚠️  Don't forget to add videos to ../videos directory!")

    print("\n" + "=" * 60 + "\n")

    # Run the server
    try:
        uvicorn.run(
            "rope_api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
        print("🧹 Cleaning up...")

        # Final cleanup
        if not args.keep_faces:
            clear_old_faces()

        print("👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()