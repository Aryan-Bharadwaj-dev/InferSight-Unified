from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

# Import deepfake detection components
from models import DetectionResult, AnalysisProgress, UploadResponse
from detection_service import DetectionService
from file_handler import FileHandler


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize services
detection_service = DetectionService()
file_handler = FileHandler()

# Create the main app without a prefix
app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Deepfake Detection API", "version": "1.0.0"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Deepfake Detection Endpoints

@api_router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a file for deepfake detection"""
    try:
        # Save uploaded file
        file_info = await file_handler.save_upload_file(file)
        
        # Start analysis in background
        upload_id = await detection_service.start_analysis(file_info)
        
        # Schedule cleanup of old files
        background_tasks.add_task(file_handler.cleanup_old_files)
        background_tasks.add_task(detection_service.cleanup_completed_analyses)
        
        return UploadResponse(
            upload_id=upload_id,
            message="File uploaded successfully. Analysis started."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing upload")

@api_router.get("/analysis/{upload_id}", response_model=AnalysisProgress)
async def get_analysis_progress(upload_id: str):
    """Get the progress of an ongoing analysis"""
    try:
        progress = await detection_service.get_analysis_progress(upload_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis progress: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analysis progress")

@api_router.get("/result/{upload_id}", response_model=DetectionResult)
async def get_detection_result(upload_id: str):
    """Get the final detection result"""
    try:
        progress = await detection_service.get_analysis_progress(upload_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if progress.status == "processing":
            raise HTTPException(status_code=202, detail="Analysis still in progress")
        
        if progress.status == "failed":
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        if not progress.result:
            raise HTTPException(status_code=404, detail="Result not available")
        
        # Store result in database
        await db.detection_results.insert_one(progress.result.dict())
        
        return progress.result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detection result: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving result")

@api_router.get("/history", response_model=List[DetectionResult])
async def get_detection_history(limit: int = 10):
    """Get recent detection results"""
    try:
        results = await db.detection_results.find().sort("created_at", -1).limit(limit).to_list(limit)
        return [DetectionResult(**result) for result in results]
        
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving history")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        await db.command("ping")
        
        return {
            "status": "healthy",
            "database": "connected",
            "services": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

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

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
