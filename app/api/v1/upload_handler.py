import magic
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from app.services.model_manager import ModelManager
from app.utils.secure_file_uploader import SecureFileUploader
from app.core.secure_config import SecureConfig
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize secure file uploader
uploader = SecureFileUploader("/app/uploads")

@router.post("/upload-image")
async def secure_upload(file: UploadFile = File(...)):
    """
    Secure file upload endpoint with comprehensive validation
    
    Returns:
        AI analysis results for the uploaded image
    """
    try:
        # Use secure file uploader to validate and save file
        file_path = await uploader.save_upload(file)
        
        # Read file content for AI processing
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Process with AI models
        model_manager = ModelManager()
        await model_manager.initialize()  # Ensure models are loaded
        results = await model_manager.analyze_face_pipeline(file_content)
        
        # Add file path to results
        results["file_path"] = str(file_path)
        
        return results
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
