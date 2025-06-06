import magic
from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile, HTTPException
import os
import aiofiles
import logging

logger = logging.getLogger(__name__)

class SecureFileUploader:
    """Secure file upload handler"""
    
    def __init__(self, upload_dir: str, max_size_bytes: int = 10 * 1024 * 1024):
        """
        Initialize secure file uploader
        
        Args:
            upload_dir: Path to upload directory
            max_size_bytes: Maximum allowed file size in bytes (default: 10MB)
        """
        self.upload_dir = Path(upload_dir).resolve()
        self.max_size_bytes = max_size_bytes
        self.allowed_types = ['image/jpeg', 'image/png', 'image/gif']
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def save_upload(self, file: UploadFile) -> Path:
        """
        Securely save uploaded file
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Path to saved file
            
        Raises:
            HTTPException: If file is invalid or too large
        """
        # 1. Validate file size BEFORE processing
        file_size = await self._get_file_size(file)
        if file_size > self.max_size_bytes:
            raise HTTPException(400, f"File too large. Maximum size is {self.max_size_bytes / (1024 * 1024)}MB")
        
        # 2. Validate file type using magic bytes
        file_content = await file.read()
        file_type = self._get_file_type(file_content)
        
        if file_type not in self.allowed_types:
            raise HTTPException(400, f"Invalid file type. Allowed types: {', '.join(self.allowed_types)}")
        
        # 3. Generate secure filename
        safe_filename = f"{uuid4()}.{file_type.split('/')[-1]}"
        
        # 4. Use absolute path validation
        file_path = (self.upload_dir / safe_filename).resolve()
        
        # 5. Ensure path is within allowed directory
        if not str(file_path).startswith(str(self.upload_dir)):
            raise HTTPException(400, "Invalid file path")
        
        # 6. Save file
        await self._save_file(file_path, file_content)
        
        logger.info(f"File saved securely: {file_path}")
        return file_path
    
    async def _get_file_size(self, file: UploadFile) -> int:
        """Get file size in bytes"""
        # Seek to end of file to get size
        await file.seek(0, 2)  # 2 means end of file
        size = file.tell()
        await file.seek(0)  # Reset position
        return size
    
    def _get_file_type(self, content: bytes) -> str:
        """Get file type from content using magic bytes"""
        mime = magic.Magic(mime=True)
        return mime.from_buffer(content)
    
    async def _save_file(self, path: Path, content: bytes) -> None:
        """Save file to disk"""
        async with aiofiles.open(path, 'wb') as f:
            await f.write(content)
