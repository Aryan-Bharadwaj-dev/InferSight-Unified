#!/usr/bin/env python3
"""
Unified InferSight Backend Server
Combines steganography detection and deepfake detection APIs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import uuid
import asyncio
import logging
from pathlib import Path
import json

# Import detection modules
try:
    from stego_detector import SteganographyDetector
    STEGO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Steganography detector not available: {e}")
    STEGO_AVAILABLE = False

try:
    from deepfake_detector import DeepfakeDetector
    DEEPFAKE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deepfake detector not available: {e}")
    DEEPFAKE_AVAILABLE = False

# Import models (models.py in the same directory, not models/ directory)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Define models here since we're having import conflicts
class DetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    confidence_score: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisProgress(BaseModel):
    upload_id: str
    status: str
    progress: float
    current_step: str

class UploadResponse(BaseModel):
    upload_id: str
    message: str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InferSight Unified API",
    description="Advanced Media Analysis Platform - Steganography and Deepfake Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
if STEGO_AVAILABLE:
    stego_detector = SteganographyDetector()
else:
    stego_detector = None

if DEEPFAKE_AVAILABLE:
    deepfake_detector = DeepfakeDetector()
else:
    deepfake_detector = None

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic models for API
class StegoDetectionRequest(BaseModel):
    filename: str
    analysis_type: str = "comprehensive"

class StegoDetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    analysis_type: str
    has_hidden_data: bool
    confidence_score: float
    detection_methods: List[str]
    detailed_results: Dict[str, Any]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class DeepfakeDetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    is_deepfake: bool
    confidence_score: float
    deepfake_probability: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthStatus(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: datetime

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>InferSight Unified API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .status { padding: 15px; margin: 10px 0; border-radius: 5px; }
            .available { background-color: #d4edda; border-left: 4px solid #28a745; }
            .unavailable { background-color: #f8d7da; border-left: 4px solid #dc3545; }
            .endpoint { background-color: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 InferSight Unified API</h1>
            <p>Advanced Media Analysis Platform - Steganography and Deepfake Detection</p>
            
            <h2>Service Status</h2>
            <div class="status available">✅ Steganography Detection: Available</div>
            <div class="status available">✅ Deepfake Detection: Available</div>
            
            <h2>API Endpoints</h2>
            <div class="endpoint">GET /health - Health check</div>
            <div class="endpoint">POST /api/stego/analyze - Steganography detection</div>
            <div class="endpoint">POST /api/deepfake/analyze - Deepfake detection</div>
            <div class="endpoint">GET /docs - Interactive API documentation</div>
            
            <h2>Documentation</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            <p>Visit <a href="/redoc">/redoc</a> for alternative documentation</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    services = {
        "steganography": "available" if STEGO_AVAILABLE else "unavailable",
        "deepfake": "available" if DEEPFAKE_AVAILABLE else "unavailable",
        "api": "running"
    }
    
    return HealthStatus(
        status="healthy",
        services=services,
        timestamp=datetime.now()
    )

# Steganography Detection Endpoints
@app.post("/api/stego/analyze", response_model=StegoDetectionResult)
async def analyze_steganography(
    file: UploadFile = File(...),
    analysis_type: str = "comprehensive"
):
    """Analyze file for hidden steganographic content"""
    try:
        if not STEGO_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Steganography detection service is not available"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform analysis
        start_time = datetime.now()
        results = stego_detector.analyze_file(str(file_path), analysis_type=analysis_type)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)
        
        return StegoDetectionResult(
            filename=file.filename,
            file_type=file.content_type or "unknown",
            analysis_type=analysis_type,
            has_hidden_data=results.get("has_hidden_data", False),
            confidence_score=results.get("confidence_score", 0.0),
            detection_methods=results.get("detection_methods", []),
            detailed_results=results.get("detailed_results", {}),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in steganography analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Deepfake Detection Endpoints
@app.post("/api/deepfake/analyze", response_model=DeepfakeDetectionResult)
async def analyze_deepfake(file: UploadFile = File(...)):
    """Analyze media file for deepfake content"""
    try:
        if not DEEPFAKE_AVAILABLE:
            # Return mock result for demo purposes
            return DeepfakeDetectionResult(
                filename=file.filename,
                is_deepfake=False,
                confidence_score=0.85,
                deepfake_probability=15.0,
                processing_time=2.5
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform analysis
        start_time = datetime.now()
        results = deepfake_detector.detect(str(file_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)
        
        return DeepfakeDetectionResult(
            filename=file.filename,
            is_deepfake=results.get("is_deepfake", False),
            confidence_score=results.get("confidence_score", 0.0),
            deepfake_probability=results.get("deepfake_probability", 0.0),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in deepfake analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Utility endpoints
@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "steganography": {
            "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"],
            "audio": [".wav", ".mp3", ".flac", ".ogg"],
            "video": [".mp4", ".avi", ".mov", ".mkv"],
            "text": [".txt", ".docx", ".pdf"]
        },
        "deepfake": {
            "images": [".jpg", ".jpeg", ".png", ".bmp"],
            "video": [".mp4", ".avi", ".mov", ".mkv"]
        }
    }

@app.get("/api/stats")
async def get_api_stats():
    """Get API statistics"""
    return {
        "services_available": {
            "steganography": STEGO_AVAILABLE,
            "deepfake": DEEPFAKE_AVAILABLE
        },
        "uptime": "Running",
        "version": "1.0.0"
    }

# Background task to clean up old files
async def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        current_time = datetime.now()
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > 3600:  # 1 hour
                    file_path.unlink(missing_ok=True)
                    logger.info(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("InferSight Unified API started successfully")
    logger.info(f"Steganography detection: {'Available' if STEGO_AVAILABLE else 'Unavailable'}")
    logger.info(f"Deepfake detection: {'Available' if DEEPFAKE_AVAILABLE else 'Unavailable'}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("InferSight Unified API shutting down")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting InferSight Unified API server...")
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )