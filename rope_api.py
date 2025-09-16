import asyncio
import uuid
import base64
from pathlib import Path
from typing import List
from datetime import datetime
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

# Import the CLI module
from rope_cli import RopeCLI

# Configuration
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("../output")
FACES_DIR = Path("../faces")
VIDEOS_DIR = Path("../videos")
TEMP_DIR = Path("./temp")
STATIC_DIR = Path("./static")
TEMPLATES_DIR = Path("./templates")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, FACES_DIR, VIDEOS_DIR, TEMP_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global processing queue and status tracker
processing_queue = asyncio.Queue()
processing_status = {}
websocket_connections = []


# Utility function to clear faces directory
def clear_faces_directory():
    """Clear all files in the faces directory"""
    print(f"🧹 Clearing faces directory: {FACES_DIR}")
    if FACES_DIR.exists():
        count = 0
        for file in FACES_DIR.glob("*"):
            if file.is_file():
                file.unlink()
                count += 1
        print(f"   Deleted {count} file(s)")
        return True
    return False


# Background worker for processing videos
async def process_video_worker():
    """Background worker that processes videos from the queue"""
    print("🚀 Background worker started and ready to process videos")

    while True:
        try:
            # Check if there are jobs in queue
            if not processing_queue.empty():
                print(f"📋 Queue has {processing_queue.qsize()} job(s) waiting")

            # Get job from queue (this will wait if queue is empty)
            job = await processing_queue.get()
            job_id = job['job_id']

            print(f"\n{'=' * 60}")
            print(f"🎬 Starting job: {job_id}")
            print(f"   Video: {job['video_file']}")
            print(f"   Quality: CRF={job.get('quality', 14)}")
            print(f"   Preset: {job.get('preset', 'slow')}")
            print(f"{'=' * 60}\n")

            # Update status
            processing_status[job_id]['status'] = 'processing'
            processing_status[job_id]['message'] = 'Starting video processing...'
            processing_status[job_id]['progress'] = 0.0

            # Notify websockets if connected
            await notify_websockets(job_id, processing_status[job_id])

            try:
                # Get the running event loop from the main thread before starting the executor
                main_loop = asyncio.get_running_loop()

                # Create progress callback for real-time updates
                async def progress_callback(progress: float, message: str):
                    """Callback to update progress through WebSocket"""
                    processing_status[job_id]['progress'] = min(progress, 99.0)  # Cap at 99% until fully complete
                    processing_status[job_id]['message'] = message
                    await notify_websockets(job_id, processing_status[job_id])
                    # Small delay to prevent flooding
                    await asyncio.sleep(0.05)

                # Create a thread-safe wrapper for the async progress callback
                def sync_progress_wrapper(progress: float, message: str):
                    """
                    Sync wrapper that safely calls the async progress callback
                    from a different thread.
                    """
                    asyncio.run_coroutine_threadsafe(
                        progress_callback(progress, message),
                        main_loop
                    )

                # Create CLI instance with the thread-safe progress callback
                print("📦 Initializing Rope CLI with progress tracking...")
                cli = RopeCLI(progress_callback=sync_progress_wrapper)

                # Load faces
                faces_path = FACES_DIR
                print(f"👤 Loading faces from: {faces_path}")
                cli.load_source_faces(str(faces_path))

                # Find faces in video
                video_path = VIDEOS_DIR / job['video_file']
                if not video_path.exists():
                    video_path = UPLOAD_DIR / job['video_file']

                print(f"🔍 Finding faces in video: {video_path}")
                num_faces = cli.find_faces_in_video(str(video_path))

                if num_faces == 0:
                    raise ValueError("No faces found in the video")

                # Process video with progress callback
                print(f"🎥 Processing video with deepfake (main actor only)...")
                # Run the blocking `process_video` function in a separate thread
                output_file = await main_loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    cli.process_video,
                    str(video_path),
                    str(OUTPUT_DIR),
                    job.get('quality', 14),
                    job.get('threads', 4),
                    job.get('codec', 'libx264'),
                    job.get('preset', 'slow')
                    # We don't need to pass the callback here since it's set during CLI initialization
                )

                # Update status to completed
                processing_status[job_id]['status'] = 'completed'
                processing_status[job_id]['progress'] = 100.0
                processing_status[job_id]['message'] = 'Processing completed successfully!'
                processing_status[job_id]['output_file'] = Path(output_file).name
                processing_status[job_id]['completed_at'] = datetime.now()

                await notify_websockets(job_id, processing_status[job_id])

                print(f"✅ Job {job_id} completed successfully!")
                print(f"   Output: {output_file}\n")

            except Exception as e:
                error_msg = f'Processing error: {str(e)}'
                print(f"❌ Job {job_id} failed: {error_msg}")

                processing_status[job_id]['status'] = 'failed'
                processing_status[job_id]['message'] = error_msg
                processing_status[job_id]['completed_at'] = datetime.now()

                await notify_websockets(job_id, processing_status[job_id])

                import traceback
                traceback.print_exc()

        except asyncio.CancelledError:
            print("⚠️ Worker cancelled")
            break
        except Exception as e:
            print(f"❌ Worker error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


async def notify_websockets(job_id: str, status: dict):
    """Notify all connected websockets about status updates"""
    for ws in websocket_connections[:]:  # Use slice to avoid modification during iteration
        try:
            await ws.send_json({
                "type": "status_update",
                "job_id": job_id,
                "status": status
            })
        except:
            # Remove disconnected websockets
            if ws in websocket_connections:
                websocket_connections.remove(ws)


# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n🌟 Starting Rope Deepfake API Server...")
    print("📌 Initializing background worker...")

    # Create and start the background worker task
    worker_task = asyncio.create_task(process_video_worker())

    print("✅ Server ready to accept requests!\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Rope Deepfake API",
    description="Web API for Rope deepfake with WebCam support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# API Endpoints

@app.get("/")
async def root(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/videos")
async def list_videos():
    """List all available videos from ../videos directory"""
    videos = []

    if VIDEOS_DIR.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            for video_file in VIDEOS_DIR.glob(ext):
                videos.append({
                    'filename': video_file.name,
                    'size': video_file.stat().st_size,
                    'modified': datetime.fromtimestamp(video_file.stat().st_mtime).isoformat()
                })

    print(f"📹 Found {len(videos)} videos in {VIDEOS_DIR}")

    return {"videos": videos}


@app.post("/api/webcam/capture")
async def capture_face(frames_data: List[str] = Form(...)):
    """Capture multiple frames from webcam and save to ../faces directory"""
    try:
        print(f"📸 Receiving {len(frames_data)} webcam frames...")

        # Clear existing faces first
        clear_faces_directory()

        # Process and save each frame
        saved_count = 0
        for i, frame_base64 in enumerate(frames_data):
            try:
                # Parse base64 image (remove data:image/jpeg;base64, prefix if present)
                if ',' in frame_base64:
                    frame_base64 = frame_base64.split(',')[1]

                image_bytes = base64.b64decode(frame_base64)

                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Save to faces directory with sequential naming
                    filename = f"face_{i:03d}.jpg"
                    filepath = FACES_DIR / filename

                    cv2.imwrite(str(filepath), img)
                    saved_count += 1
                    print(f"   ✓ Saved: {filename}")
                else:
                    print(f"   ⚠ Failed to decode frame {i}")

            except Exception as e:
                print(f"   ⚠ Error processing frame {i}: {e}")
                continue

        if saved_count == 0:
            raise ValueError("No valid frames could be saved")

        print(f"✅ Successfully saved {saved_count} face frames to {FACES_DIR}")

        return {
            "success": True,
            "filename": f"face_000.jpg",  # Return first filename for compatibility
            "message": f"Captured {saved_count} face frames successfully",
            "faces_count": saved_count
        }

    except Exception as e:
        print(f"❌ Error capturing faces: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/faces/count")
async def get_face_count():
    """Get count of faces in ../faces directory"""
    count = len(list(FACES_DIR.glob("*.jpg"))) + len(list(FACES_DIR.glob("*.png")))
    print(f"👤 Face count: {count}")
    return {"faces_count": count}


@app.post("/api/faces/clear")
async def clear_faces():
    """Clear all faces from ../faces directory"""
    if clear_faces_directory():
        return {"success": True, "message": "All faces cleared"}
    return {"success": False, "message": "No faces to clear"}


@app.post("/api/process/deepfake")
async def process_deepfake(
        video_file: str = Form(...),
        quality: int = Form(14),
        threads: int = Form(4),
        codec: str = Form("libx264"),
        preset: str = Form("slow")
):
    """Queue a deepfake processing job"""
    try:
        print(f"\n📝 New deepfake request:")
        print(f"   Video: {video_file}")
        print(f"   Quality: {quality}")
        print(f"   Threads: {threads}")
        print(f"   Preset: {preset}")

        # Check if faces exist
        face_count = len(list(FACES_DIR.glob("*.jpg"))) + len(list(FACES_DIR.glob("*.png")))
        if face_count == 0:
            raise HTTPException(status_code=400, detail="No source faces available. Please capture faces first.")

        print(f"   Faces available: {face_count}")

        # Create job
        job_id = uuid.uuid4().hex
        job = {
            'job_id': job_id,
            'video_file': video_file,
            'quality': quality,
            'threads': threads,
            'codec': codec,
            'preset': preset
        }

        # Initialize status
        processing_status[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0.0,
            'message': 'Job queued for processing',
            'output_file': None,
            'created_at': datetime.now(),
            'completed_at': None
        }

        # Add to queue
        await processing_queue.put(job)
        print(f"✅ Job {job_id} added to queue (queue size: {processing_queue.qsize()})")

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Deepfake processing queued (will swap main actor only)"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error starting deepfake processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return processing_status[job_id]


@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Download the processed video"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    print(f"📥 Serving download: {filename}")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    print(f"🔌 WebSocket connected (total: {len(websocket_connections)})")

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        print(f"🔌 WebSocket disconnected (remaining: {len(websocket_connections)})")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_size = processing_queue.qsize()
    active_jobs = len([j for j in processing_status.values() if j['status'] == 'processing'])

    return {
        "status": "healthy",
        "version": "2.0.0",
        "queue_size": queue_size,
        "active_jobs": active_jobs,
        "websocket_connections": len(websocket_connections)
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🎭 ROPE DEEPFAKE API - Starting...")
    print("=" * 60 + "\n")

    print(f"📁 Checking directories:")
    print(f"   Videos: {VIDEOS_DIR.absolute()}")
    print(f"   Faces: {FACES_DIR.absolute()}")
    print(f"   Output: {OUTPUT_DIR.absolute()}")

    # Check if videos directory exists and has videos
    if VIDEOS_DIR.exists():
        video_count = len(list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.avi")) +
                          list(VIDEOS_DIR.glob("*.mov")) + list(VIDEOS_DIR.glob("*.mkv")))
        print(f"   Found {video_count} video(s) in {VIDEOS_DIR}")
        if video_count == 0:
            print(f"   ⚠️ No videos found! Please add video files to: {VIDEOS_DIR.absolute()}")
    else:
        print(f"   ⚠️ Videos directory doesn't exist! Creating: {VIDEOS_DIR.absolute()}")
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        "rope_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
