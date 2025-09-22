from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class DetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str  # 'image', 'video', 'audio'
    file_size: int
    content_type: str
    
    # Detection results
    is_deepfake: bool
    confidence_score: float  # 0.0 to 1.0
    deepfake_probability: float  # percentage (0-100)
    
    # Detailed analysis
    visual_analysis: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # Model information
    models_used: List[str]
    processing_time: float
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "completed"  # "processing", "completed", "failed"
    error_message: Optional[str] = None

class DetectionRequest(BaseModel):
    filename: str
    file_type: str
    content_type: str
    file_size: int

class UploadResponse(BaseModel):
    upload_id: str
    message: str

class AnalysisProgress(BaseModel):
    upload_id: str
    status: str  # "uploading", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_step: str
    estimated_time_remaining: Optional[float] = None
    result: Optional[DetectionResult] = None