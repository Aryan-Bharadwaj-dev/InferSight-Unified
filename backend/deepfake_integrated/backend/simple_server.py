from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio
import time

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app
app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Simple in-memory storage for demo
active_analyses = {}

# Define Models
class UploadResponse(BaseModel):
    upload_id: str
    message: str

class DetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    file_size: int
    content_type: str
    is_deepfake: bool
    confidence_score: float
    deepfake_probability: float
    models_used: List[str]
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "completed"
    error_message: Optional[str] = None

class AnalysisProgress(BaseModel):
    upload_id: str
    status: str
    progress: float
    current_step: str
    estimated_time_remaining: Optional[float] = None
    result: Optional[DetectionResult] = None

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Deepfake Detection API", "version": "1.0.0"}

@api_router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a file for deepfake detection"""
    try:
        # Generate upload ID
        upload_id = str(uuid.uuid4())
        
        # Initialize progress
        progress = AnalysisProgress(
            upload_id=upload_id,
            status="processing",
            progress=0.0,
            current_step="File uploaded, starting analysis"
        )
        active_analyses[upload_id] = progress
        
        # Start mock analysis in background
        background_tasks.add_task(mock_analysis, upload_id, file)
        
        return UploadResponse(
            upload_id=upload_id,
            message="File uploaded successfully. Analysis started."
        )
        
    except Exception as e:
        logging.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing upload")

@api_router.get("/analysis/{upload_id}", response_model=AnalysisProgress)
async def get_analysis_progress(upload_id: str):
    """Get the progress of an ongoing analysis"""
    try:
        progress = active_analyses.get(upload_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting analysis progress: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analysis progress")

@api_router.get("/result/{upload_id}", response_model=DetectionResult)
async def get_detection_result(upload_id: str):
    """Get the final detection result"""
    try:
        progress = active_analyses.get(upload_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if progress.status == "processing":
            raise HTTPException(status_code=202, detail="Analysis still in progress")
        
        if progress.status == "failed":
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        if not progress.result:
            raise HTTPException(status_code=404, detail="Result not available")
        
        return progress.result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting detection result: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving result")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "simulated",
        "services": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

async def mock_analysis(upload_id: str, file: UploadFile):
    """Mock analysis function for testing"""
    try:
        progress = active_analyses[upload_id]
        
        # Simulate analysis steps
        steps = [
            (0.1, "Preparing analysis"),
            (0.3, "Analyzing image for facial manipulations"),
            (0.5, "Processing neural network models"),
            (0.7, "Performing deepfake detection"),
            (0.9, "Finalizing analysis"),
            (1.0, "Analysis completed")
        ]
        
        for target_progress, step in steps:
            progress.progress = target_progress
            progress.current_step = step
            await asyncio.sleep(1)  # Simulate processing time
        
        # Generate mock result
        import random
        is_deepfake = random.choice([True, False])
        confidence = random.uniform(0.6, 0.95)
        
        result = DetectionResult(
            filename=file.filename or "unknown",
            file_type="image" if file.content_type.startswith("image") else "video",
            file_size=0,  # We don't actually save the file
            content_type=file.content_type or "application/octet-stream",
            is_deepfake=is_deepfake,
            confidence_score=confidence,
            deepfake_probability=confidence * 100,
            models_used=["Mock Detection Model v1.0"],
            processing_time=len(steps) * 1.0
        )
        
        progress.result = result
        progress.status = "completed"
        
        logging.info(f"Mock analysis completed for {upload_id}")
        
    except Exception as e:
        logging.error(f"Error in mock analysis {upload_id}: {e}")
        if upload_id in active_analyses:
            active_analyses[upload_id].status = "failed"
            active_analyses[upload_id].error_message = str(e)

# Static file serving for frontend
static_dir = ROOT_DIR.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Root route to serve the main deepfake page
@app.get("/")
async def serve_index():
    return FileResponse(str(static_dir / "deepfake.html"))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
