import os
import aiofiles
import tempfile
import shutil
from typing import Dict, Any, Optional
from fastapi import UploadFile, HTTPException
import uuid
import mimetypes
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file uploads and temporary storage"""
    
    def __init__(self):
        self.upload_dir = "/tmp/deepfake_uploads"
        self.max_file_size = 2 * 1024 * 1024  # 2MB limit
        self.allowed_extensions = {
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
            'audio': ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
        }
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Create upload directory if it doesn't exist"""
        try:
            os.makedirs(self.upload_dir, exist_ok=True)
            logger.info(f"Upload directory ready: {self.upload_dir}")
        except Exception as e:
            logger.error(f"Error creating upload directory: {e}")
            raise
    
    async def save_upload_file(self, upload_file: UploadFile) -> Dict[str, Any]:
        """Save uploaded file and return file info"""
        try:
            # Validate file size
            file_content = await upload_file.read()
            file_size = len(file_content)
            
            if file_size > self.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size ({file_size} bytes) exceeds limit ({self.max_file_size} bytes)"
                )
            
            # Determine file type and validate extension
            file_extension = os.path.splitext(upload_file.filename.lower())[1]
            file_type = self._get_file_type(file_extension, upload_file.content_type)
            
            if not file_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )
            
            # Generate unique filename
            upload_id = str(uuid.uuid4())
            safe_filename = f"{upload_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            file_info = {
                'upload_id': upload_id,
                'filename': upload_file.filename,
                'safe_filename': safe_filename,
                'file_path': file_path,
                'file_type': file_type,
                'file_size': file_size,
                'content_type': upload_file.content_type,
                'file_extension': file_extension
            }
            
            logger.info(f"File saved: {upload_file.filename} -> {safe_filename}")
            return file_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Error saving file")
    
    def _get_file_type(self, extension: str, content_type: str) -> Optional[str]:
        """Determine file type from extension and content type"""
        # Check by extension first
        for file_type, extensions in self.allowed_extensions.items():
            if extension in extensions:
                return file_type
        
        # Check by content type
        if content_type:
            if content_type.startswith('image/'):
                return 'image'
            elif content_type.startswith('video/'):
                return 'video'
            elif content_type.startswith('audio/'):
                return 'audio'
        
        return None
    
    def cleanup_file(self, file_path: str):
        """Remove temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary files"""
        try:
            import time
            current_time = time.time()
            
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                file_age = current_time - os.path.getctime(file_path)
                
                if file_age > max_age_hours * 3600:  # Convert hours to seconds
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")