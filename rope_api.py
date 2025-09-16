import asyncio
import uuid
import base64
from pathlib import Path
from typing import List, Optional
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
from pydantic import BaseModel
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
            processing_status[job_id]['progress'] = 5.0

            # Notify websockets if connected
            await notify_websockets(job_id, processing_status[job_id])

            try:
                # Create CLI instance
                print("📦 Initializing Rope CLI...")
                cli = RopeCLI()

                # Load faces
                faces_path = FACES_DIR
                print(f"👤 Loading faces from: {faces_path}")
                cli.load_source_faces(str(faces_path))

                processing_status[job_id]['message'] = 'Loaded source faces...'
                processing_status[job_id]['progress'] = 15.0
                await notify_websockets(job_id, processing_status[job_id])

                # Find faces in video
                video_path = VIDEOS_DIR / job['video_file']
                if not video_path.exists():
                    video_path = UPLOAD_DIR / job['video_file']

                print(f"🔍 Finding faces in video: {video_path}")
                num_faces = cli.find_faces_in_video(str(video_path))

                processing_status[job_id]['message'] = f'Found {num_faces} faces. Processing...'
                processing_status[job_id]['progress'] = 25.0
                await notify_websockets(job_id, processing_status[job_id])

                # Process video
                print(f"🎥 Processing video with deepfake...")
                output_file = cli.process_video(
                    str(video_path),
                    str(OUTPUT_DIR),
                    quality=job.get('quality', 14),
                    threads=job.get('threads', 4),
                    codec=job.get('codec', 'libx264'),
                    preset=job.get('preset', 'slow')
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


# Pydantic models
class ProcessingRequest(BaseModel):
    video_file: str
    quality: int = 14
    threads: int = 4
    codec: str = "libx264"
    preset: str = "slow"


class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    output_file: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# Utility functions
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
                    "filename": video_file.name,
                    "size": video_file.stat().st_size,
                    "modified": datetime.fromtimestamp(video_file.stat().st_mtime)
                })

    print(f"📹 Found {len(videos)} video(s) in {VIDEOS_DIR}")
    return {"videos": videos, "count": len(videos)}


@app.post("/api/webcam/capture")
async def capture_webcam_frames(frames_data: List[str] = Form(...)):
    """Capture frames from webcam and save to faces directory"""
    try:
        print(f"📸 Receiving {len(frames_data)} webcam frames...")

        # Clear existing faces first
        clear_faces_directory()

        # Process and save frames
        saved_count = 0
        for i, frame_base64 in enumerate(frames_data):
            try:
                # Decode base64 image
                img_data = base64.b64decode(frame_base64.split(',')[1])

                # Convert to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Save to faces directory
                    filename = f"face_{i:03d}.jpg"
                    filepath = FACES_DIR / filename
                    cv2.imwrite(str(filepath), img)
                    saved_count += 1
            except Exception as e:
                print(f"   Warning: Error processing frame {i}: {e}")
                continue

        print(f"✅ Saved {saved_count} face frames to {FACES_DIR}")

        return {
            "success": True,
            "message": f"Saved {saved_count} face frames",
            "faces_count": saved_count
        }

    except Exception as e:
        print(f"❌ Error capturing webcam frames: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faces/clear")
async def clear_faces():
    """Clear all faces from the faces directory"""
    try:
        success = clear_faces_directory()

        if success:
            return {
                "success": True,
                "message": "All faces cleared successfully"
            }
        else:
            return {
                "success": False,
                "message": "Faces directory not found"
            }

    except Exception as e:
        print(f"❌ Error clearing faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/faces/count")
async def count_faces():
    """Count the number of face images in the faces directory"""
    count = 0
    if FACES_DIR.exists():
        count = len(list(FACES_DIR.glob("*.jpg")) +
                    list(FACES_DIR.glob("*.png")) +
                    list(FACES_DIR.glob("*.jpeg")))

    print(f"👤 Face count: {count}")
    return {"faces_count": count}


@app.post("/api/process/deepfake")
async def start_deepfake_processing(
        video_file: str = Form(...),
        quality: int = Form(14),
        threads: int = Form(4),
        codec: str = Form("libx264"),
        preset: str = Form("slow")
):
    """Start deepfake processing"""
    try:
        print(f"\n🎯 New deepfake request received:")
        print(f"   Video: {video_file}")
        print(f"   Quality: CRF={quality}")
        print(f"   Threads: {threads}")
        print(f"   Preset: {preset}")

        # Check if video exists
        video_path = VIDEOS_DIR / video_file
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            raise HTTPException(status_code=404, detail="Video not found")

        # Check if we have faces
        face_count = len(list(FACES_DIR.glob("*")))
        if face_count == 0:
            print(f"❌ No face images found in {FACES_DIR}")
            raise HTTPException(status_code=400, detail="No face images found. Please capture faces first.")

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
            "message": "Deepfake processing queued"
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


@app.get("/api/jobs")
async def list_all_jobs():
    """List all processing jobs"""
    jobs = list(processing_status.values())

    active = len([j for j in jobs if j['status'] == 'processing'])
    completed = len([j for j in jobs if j['status'] == 'completed'])
    queued = len([j for j in jobs if j['status'] == 'queued'])
    failed = len([j for j in jobs if j['status'] == 'failed'])

    print(f"📊 Jobs - Total: {len(jobs)}, Active: {active}, Queued: {queued}, Completed: {completed}, Failed: {failed}")

    return {
        "jobs": jobs,
        "total": len(jobs),
        "active": active,
        "queued": queued,
        "completed": completed,
        "failed": failed
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_size = processing_queue.qsize()
    active_jobs = len([j for j in processing_status.values() if j['status'] == 'processing'])

    return {
        "status": "healthy",
        "version": "2.0.0",
        "queue_size": queue_size,
        "active_jobs": active_jobs
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🎭 ROPE DEEPFAKE API - Starting...")
    print("=" * 60 + "\n")

    uvicorn.run(
        "rope_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )