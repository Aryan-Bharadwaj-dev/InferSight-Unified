#!/usr/bin/env python3
"""
FastAPI backend for the Steganography Detection Tool
Provides REST API endpoints for detecting steganography in various file types
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import Form
import tempfile
import os
import shutil
import uuid
from pathlib import Path
import asyncio
from datetime import datetime
import json

# Import our detection modules
from stego_detector import SteganographyDetector
from utils.report_generator import ReportGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Steganography Detection API",
    description="API for detecting steganography in images, videos, audio files, and text files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and report generator
detector = SteganographyDetector()
report_generator = ReportGenerator()

# In-memory storage for analysis results (use database in production)
analysis_results = {}

# Pydantic models for API responses
class DetectionResult(BaseModel):
    method: str
    confidence: float
    description: str
    details: Optional[Dict[str, Any]] = None

class FileAnalysisResult(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    file_size: int
    status: str
    detections: List[DetectionResult]
    analysis_time: Optional[str] = None
    error: Optional[str] = None

class AnalysisStatus(BaseModel):
    file_id: str
    status: str
    message: str

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'image': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'},
    'audio': {'.wav', '.mp3', '.flac', '.ogg', '.aac'},
    'video': {'.mp4', '.avi', '.mkv', '.mov', '.wmv'},
    'text': {'.txt', '.md', '.html', '.xml'}
}

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = Path(filename).suffix.lower()
    for file_type, extensions in SUPPORTED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return 'unknown'

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported"""
    return get_file_type(filename) != 'unknown'

async def analyze_file_async(file_id: str, file_path: Path, original_filename: str):
    """Asynchronously analyze a file for steganography"""
    try:
        # Update status to processing
        analysis_results[file_id]['status'] = 'processing'
        analysis_results[file_id]['message'] = 'Analyzing file for steganography...'
        
        # Run detection
        result = detector.detect_file(file_path)
        
        if result:
            # Convert detection results to Pydantic models
            detections = [
                DetectionResult(
                    method=d['method'],
                    confidence=d['confidence'],
                    description=d['description'],
                    details=d.get('details')
                ) for d in result.get('detections', [])
            ]
            
            # Update results
            analysis_results[file_id].update({
                'status': 'completed',
                'message': 'Analysis completed successfully',
                'file_name': original_filename,
                'file_type': result['file_type'],
                'file_size': result['file_size'],
                'detections': [d.dict() for d in detections],
                'analysis_time': datetime.now().isoformat(),
                'error': result.get('error')
            })
        else:
            analysis_results[file_id].update({
                'status': 'failed',
                'message': 'Failed to analyze file',
                'error': 'Unknown error occurred'
            })
            
    except Exception as e:
        analysis_results[file_id].update({
            'status': 'failed',
            'message': f'Analysis failed: {str(e)}',
            'error': str(e)
        })
    
    finally:
        # Clean up temporary file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Steganography Detection API",
        "version": "1.0.0",
        "supported_formats": {
            "images": list(SUPPORTED_EXTENSIONS['image']),
            "audio": list(SUPPORTED_EXTENSIONS['audio']),
            "video": list(SUPPORTED_EXTENSIONS['video']),
            "text": list(SUPPORTED_EXTENSIONS['text'])
        },
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze/{file_id}",
            "status": "/status/{file_id}",
            "results": "/results/{file_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload", response_model=AnalysisStatus)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a file for steganography analysis
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    try:
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "stego_analysis"
        temp_dir.mkdir(exist_ok=True)
        
        file_extension = Path(file.filename).suffix
        temp_file_path = temp_dir / f"{file_id}{file_extension}"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Initialize analysis result
        analysis_results[file_id] = {
            'file_id': file_id,
            'status': 'queued',
            'message': 'File uploaded successfully, analysis queued',
            'original_filename': file.filename,
            'upload_time': datetime.now().isoformat()
        }
        
        # Start background analysis
        background_tasks.add_task(analyze_file_async, file_id, temp_file_path, file.filename)
        
        return AnalysisStatus(
            file_id=file_id,
            status="queued",
            message="File uploaded successfully, analysis started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/status/{file_id}", response_model=AnalysisStatus)
async def get_analysis_status(file_id: str):
    """
    Get the status of a file analysis
    """
    if file_id not in analysis_results:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    result = analysis_results[file_id]
    return AnalysisStatus(
        file_id=file_id,
        status=result['status'],
        message=result['message']
    )

@app.get("/results/{file_id}", response_model=FileAnalysisResult)
async def get_analysis_results(file_id: str):
    """
    Get the complete analysis results for a file
    """
    if file_id not in analysis_results:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    result = analysis_results[file_id]
    
    if result['status'] != 'completed':
        raise HTTPException(
            status_code=202, 
            detail=f"Analysis not completed. Current status: {result['status']}"
        )
    
    return FileAnalysisResult(
        file_id=file_id,
        file_name=result.get('file_name', ''),
        file_type=result.get('file_type', 'unknown'),
        file_size=result.get('file_size', 0),
        status=result['status'],
        detections=[DetectionResult(**d) for d in result.get('detections', [])],
        analysis_time=result.get('analysis_time'),
        error=result.get('error')
    )

@app.post("/analyze", response_model=FileAnalysisResult)
async def analyze_file_sync(file: UploadFile = File(...)):
    """
    Upload and analyze a file synchronously (for smaller files)
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    
    file_id = str(uuid.uuid4())
    
    try:
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "stego_analysis"
        temp_dir.mkdir(exist_ok=True)
        
        file_extension = Path(file.filename).suffix
        temp_file_path = temp_dir / f"{file_id}{file_extension}"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Analyze file
        result = detector.detect_file(temp_file_path)
        
        # Clean up temporary file
        temp_file_path.unlink()
        
        if result:
            detections = [
                DetectionResult(
                    method=d['method'],
                    confidence=d['confidence'],
                    description=d['description'],
                    details=d.get('details')
                ) for d in result.get('detections', [])
            ]
            
            return FileAnalysisResult(
                file_id=file_id,
                file_name=file.filename,
                file_type=result['file_type'],
                file_size=result['file_size'],
                status='completed',
                detections=detections,
                analysis_time=datetime.now().isoformat(),
                error=result.get('error')
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze file")
            
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/results/{file_id}/report")
async def download_report(file_id: str):
    """
    Download analysis report as JSON file
    """
    if file_id not in analysis_results:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    result = analysis_results[file_id]
    
    if result['status'] != 'completed':
        raise HTTPException(
            status_code=202, 
            detail=f"Analysis not completed. Current status: {result['status']}"
        )
    
    # Create temporary report file
    temp_dir = Path(tempfile.gettempdir()) / "stego_reports"
    temp_dir.mkdir(exist_ok=True)
    
    report_file = temp_dir / f"report_{file_id}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return FileResponse(
        report_file,
        media_type='application/json',
        filename=f"steganography_report_{file_id}.json"
    )

@app.delete("/results/{file_id}")
async def delete_analysis_result(file_id: str):
    """
    Delete analysis results from memory
    """
    if file_id not in analysis_results:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    del analysis_results[file_id]
    return {"message": f"Analysis result {file_id} deleted successfully"}

@app.post("/analyze-text", response_model=FileAnalysisResult)
async def analyze_text(text_content: str = Form(...)):
    """
    Analyze text content for steganography (copy-paste functionality)
    """
    if not text_content or not text_content.strip():
        raise HTTPException(status_code=400, detail="No text content provided")
    
    file_id = str(uuid.uuid4())
    
    try:
        # Create temporary text file
        temp_dir = Path(tempfile.gettempdir()) / "stego_analysis"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{file_id}.txt"
        
        # Save text content to temporary file
        with open(temp_file_path, "w", encoding='utf-8') as temp_file:
            temp_file.write(text_content)
        
        # Analyze text
        from detectors.text_detector import TextStegoDetector
        text_detector = TextStegoDetector()
        detections = text_detector.detect(temp_file_path)
        
        # Clean up temporary file
        temp_file_path.unlink()
        
        # Convert detection results
        detection_results = [
            DetectionResult(
                method=d['method'],
                confidence=d['confidence'],
                description=d['description'],
                details=d.get('details')
            ) for d in detections
        ]
        
        return FileAnalysisResult(
            file_id=file_id,
            file_name="pasted_text.txt",
            file_type="text",
            file_size=len(text_content.encode('utf-8')),
            status='completed',
            detections=detection_results,
            analysis_time=datetime.now().isoformat(),
            error=None
        )
        
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    """
    return {
        "supported_formats": SUPPORTED_EXTENSIONS,
        "total_formats": sum(len(formats) for formats in SUPPORTED_EXTENSIONS.values())
    }

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the frontend application
    """
    try:
        with open("frontend/index.html", "r", encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")

# Serve static files (frontend) - this should be last to not interfere with API routes
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except RuntimeError:
    pass  # Directory doesn't exist

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
