import os
import sys
import json
import shutil
import asyncio
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the CLI module
from rope_cli import RopeCLI

# Configuration
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
FACES_DIR = Path("./face_library")
TEMP_DIR = Path("./temp")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, FACES_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Rope Deepfake API",
    description="Web API for Rope deepfake video processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processing queue and status tracker
processing_queue = asyncio.Queue()
processing_status = {}


# Pydantic models
class ProcessingRequest(BaseModel):
    video_id: str
    faces_set: str
    quality: int = 18
    threads: int = 2


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float
    message: str
    output_file: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class FaceSet(BaseModel):
    name: str
    faces_count: int
    created_at: datetime


# Background worker for processing videos
async def process_video_worker():
    """Background worker that processes videos from the queue"""
    while True:
        try:
            job = await processing_queue.get()
            job_id = job['job_id']

            # Update status
            processing_status[job_id]['status'] = 'processing'
            processing_status[job_id]['message'] = 'Starting video processing...'

            # Create CLI instance
            cli = RopeCLI()

            try:
                # Load faces
                faces_path = FACES_DIR / job['faces_set']
                cli.load_source_faces(str(faces_path))

                # Find faces in video
                video_path = UPLOAD_DIR / job['video_file']
                cli.find_faces_in_video(str(video_path))

                # Process video
                output_file = cli.process_video(
                    str(video_path),
                    str(OUTPUT_DIR),
                    quality=job['quality'],
                    threads=job['threads']
                )

                # Update status
                processing_status[job_id]['status'] = 'completed'
                processing_status[job_id]['progress'] = 100.0
                processing_status[job_id]['message'] = 'Processing completed successfully'
                processing_status[job_id]['output_file'] = Path(output_file).name
                processing_status[job_id]['completed_at'] = datetime.now()

            except Exception as e:
                processing_status[job_id]['status'] = 'failed'
                processing_status[job_id]['message'] = f'Error: {str(e)}'
                processing_status[job_id]['completed_at'] = datetime.now()

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            await asyncio.sleep(1)


# Start background worker on app startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_video_worker())


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rope Deepfake API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload/video": "Upload a video file",
            "POST /upload/face": "Upload face images",
            "GET /faces": "List available face sets",
            "POST /process": "Start video processing",
            "GET /status/{job_id}": "Check processing status",
            "GET /download/{filename}": "Download processed video",
            "GET /jobs": "List all processing jobs"
        }
    }


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing"""
    try:
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")

        # Generate unique ID
        video_id = f"{uuid.uuid4().hex}_{file.filename}"
        video_path = UPLOAD_DIR / video_id

        # Save file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "video_id": video_id,
            "filename": file.filename,
            "size": video_path.stat().st_size,
            "message": "Video uploaded successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/face")
async def upload_face(
        face_set: str = Form(...),
        files: List[UploadFile] = File(...)
):
    """Upload face images for a face set"""
    try:
        # Create face set directory
        face_set_dir = FACES_DIR / face_set
        face_set_dir.mkdir(parents=True, exist_ok=True)

        # Save all face images
        saved_files = []
        for file in files:
            # Validate image type
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                continue

            # Save file
            file_path = face_set_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

        return {
            "face_set": face_set,
            "faces_uploaded": len(saved_files),
            "files": saved_files,
            "message": f"Face set '{face_set}' created/updated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces")
async def list_face_sets():
    """List all available face sets"""
    face_sets = []

    for face_dir in FACES_DIR.iterdir():
        if face_dir.is_dir():
            # Count face images
            face_count = len(list(face_dir.glob("*.[jp][pn][g]*")))

            face_sets.append({
                "name": face_dir.name,
                "faces_count": face_count,
                "created_at": datetime.fromtimestamp(face_dir.stat().st_ctime)
            })

    return {
        "face_sets": face_sets,
        "total": len(face_sets)
    }


@app.post("/process")
async def start_processing(
        video_id: str = Form(...),
        faces_set: str = Form(...),
        quality: int = Form(18),
        threads: int = Form(2)
):
    """Start processing a video with face swapping"""
    try:
        # Validate video exists
        video_path = UPLOAD_DIR / video_id
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        # Validate face set exists
        faces_path = FACES_DIR / faces_set
        if not faces_path.exists():
            raise HTTPException(status_code=404, detail="Face set not found")

        # Create job
        job_id = uuid.uuid4().hex
        job = {
            'job_id': job_id,
            'video_file': video_id,
            'faces_set': faces_set,
            'quality': quality,
            'threads': threads
        }

        # Add to queue
        await processing_queue.put(job)

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

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video processing job created",
            "queue_position": processing_queue.qsize()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return processing_status[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all processing jobs"""
    jobs = []
    for job_id, status in processing_status.items():
        jobs.append(status)

    return {
        "jobs": jobs,
        "total": len(jobs),
        "queued": len([j for j in jobs if j['status'] == 'queued']),
        "processing": len([j for j in jobs if j['status'] == 'processing']),
        "completed": len([j for j in jobs if j['status'] == 'completed']),
        "failed": len([j for j in jobs if j['status'] == 'failed'])
    }


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a processed video"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type='video/mp4',
        filename=filename
    )


@app.delete("/cleanup")
async def cleanup_old_files(days_old: int = 7):
    """Clean up old files from uploads and outputs"""
    from datetime import timedelta

    cutoff_time = datetime.now() - timedelta(days=days_old)
    deleted_files = []

    # Clean uploads
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                file_path.unlink()
                deleted_files.append(str(file_path))

    # Clean outputs
    for file_path in OUTPUT_DIR.iterdir():
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                file_path.unlink()
                deleted_files.append(str(file_path))

    return {
        "deleted_files": len(deleted_files),
        "files": deleted_files,
        "message": f"Cleaned up files older than {days_old} days"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "queue_size": processing_queue.qsize(),
        "active_jobs": len([j for j in processing_status.values() if j['status'] == 'processing'])
    }


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "rope_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )