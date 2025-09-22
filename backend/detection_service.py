import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from models import DetectionResult, AnalysisProgress
from deepfake_detector import DeepfakeDetector
from file_handler import FileHandler

logger = logging.getLogger(__name__)

class DetectionService:
    """Service to orchestrate deepfake detection process"""
    
    def __init__(self):
        self.detector = DeepfakeDetector()
        self.file_handler = FileHandler()
        self.active_analyses = {}  # Store progress of ongoing analyses
        
    async def start_analysis(self, file_info: Dict[str, Any]) -> str:
        """Start deepfake analysis process"""
        upload_id = file_info['upload_id']
        
        # Initialize progress tracking
        progress = AnalysisProgress(
            upload_id=upload_id,
            status="processing",
            progress=0.0,
            current_step="Initializing analysis"
        )
        self.active_analyses[upload_id] = progress
        
        # Start analysis in background
        asyncio.create_task(self._perform_analysis(file_info))
        
        return upload_id
    
    async def get_analysis_progress(self, upload_id: str) -> Optional[AnalysisProgress]:
        """Get current analysis progress"""
        return self.active_analyses.get(upload_id)
    
    async def _perform_analysis(self, file_info: Dict[str, Any]):
        """Perform the actual deepfake detection analysis"""
        upload_id = file_info['upload_id']
        file_path = file_info['file_path']
        file_type = file_info['file_type']
        
        try:
            # Update progress
            await self._update_progress(upload_id, 0.1, "Preparing analysis")
            
            # Perform detection based on file type
            if file_type == 'image':
                result_data = await self._analyze_image(file_info)
            elif file_type == 'video':
                result_data = await self._analyze_video(file_info)
            elif file_type == 'audio':
                result_data = await self._analyze_audio(file_info)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create detection result
            detection_result = DetectionResult(
                filename=file_info['filename'],
                file_type=file_type,
                file_size=file_info['file_size'],
                content_type=file_info['content_type'],
                **result_data
            )
            
            # Update progress with final result
            await self._update_progress(
                upload_id, 
                1.0, 
                "Analysis completed", 
                "completed", 
                detection_result
            )
            
            logger.info(f"Analysis completed for {upload_id}")
            
        except Exception as e:
            logger.error(f"Error in analysis {upload_id}: {e}")
            await self._update_progress(
                upload_id, 
                0.0, 
                f"Analysis failed: {str(e)}", 
                "failed"
            )
        finally:
            # Clean up uploaded file
            self.file_handler.cleanup_file(file_path)
    
    async def _analyze_image(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image for deepfake detection"""
        upload_id = file_info['upload_id']
        file_path = file_info['file_path']
        
        await self._update_progress(upload_id, 0.3, "Analyzing image for facial manipulations")
        
        result = await self.detector.detect_image(file_path)
        
        await self._update_progress(upload_id, 0.9, "Finalizing image analysis")
        
        return result
    
    async def _analyze_video(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video for deepfake detection"""
        upload_id = file_info['upload_id']
        file_path = file_info['file_path']
        
        await self._update_progress(upload_id, 0.2, "Extracting video frames")
        await asyncio.sleep(0.1)  # Allow progress update
        
        await self._update_progress(upload_id, 0.4, "Analyzing facial features")
        await asyncio.sleep(0.1)
        
        await self._update_progress(upload_id, 0.6, "Performing temporal analysis")
        await asyncio.sleep(0.1)
        
        await self._update_progress(upload_id, 0.8, "Analyzing audio track")
        
        result = await self.detector.detect_video(file_path)
        
        await self._update_progress(upload_id, 0.95, "Finalizing video analysis")
        
        return result
    
    async def _analyze_audio(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio for deepfake detection"""
        upload_id = file_info['upload_id']
        file_path = file_info['file_path']
        
        await self._update_progress(upload_id, 0.3, "Analyzing audio spectral features")
        
        result = await self.detector.detect_audio(file_path)
        
        await self._update_progress(upload_id, 0.9, "Finalizing audio analysis")
        
        return result
    
    async def _update_progress(
        self, 
        upload_id: str, 
        progress: float, 
        step: str, 
        status: str = "processing",
        result: Optional[DetectionResult] = None
    ):
        """Update analysis progress"""
        if upload_id in self.active_analyses:
            self.active_analyses[upload_id].progress = progress
            self.active_analyses[upload_id].current_step = step
            self.active_analyses[upload_id].status = status
            if result:
                self.active_analyses[upload_id].result = result
            
            logger.debug(f"Progress {upload_id}: {progress:.1%} - {step}")
    
    def cleanup_completed_analyses(self, max_age_hours: int = 1):
        """Clean up old completed analyses from memory"""
        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            to_remove = []
            for upload_id, progress in self.active_analyses.items():
                if (progress.status in ["completed", "failed"] and 
                    progress.result and 
                    progress.result.created_at < cutoff_time):
                    to_remove.append(upload_id)
            
            for upload_id in to_remove:
                del self.active_analyses[upload_id]
                logger.info(f"Cleaned up analysis: {upload_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up analyses: {e}")