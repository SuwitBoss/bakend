from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from app.services.ai_service import AIService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize AI service
ai_service = AIService()

@router.post("/detect-faces")
async def detect_faces(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Detect faces in uploaded image"""
    try:
        logger.info(f"Received file: {image.filename}, content_type: {image.content_type}")
        
        # Validate file type - more flexible validation  
        if image.content_type and not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be an image, received: {image.content_type}"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Detect faces
        result = await ai_service.detect_faces_in_image(image_bytes)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during face detection"
        )

@router.post("/analyze-deepfake")
async def analyze_deepfake(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze image for deepfake detection"""
    try:
        logger.info(f"Received file for deepfake analysis: {image.filename}, content_type: {image.content_type}")
        
        # Validate file type - more flexible validation
        if image.content_type and not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be an image, received: {image.content_type}"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Analyze for deepfake
        result = await ai_service.detect_deepfake(image_bytes)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deepfake detection API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during deepfake detection"
        )

@router.post("/analyze-antispoofing")
async def analyze_antispoofing(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze image for anti-spoofing detection"""
    try:
        logger.info(f"Received file for anti-spoofing analysis: {image.filename}, content_type: {image.content_type}")
        
        # Validate file type - more flexible validation
        if image.content_type and not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be an image, received: {image.content_type}"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Analyze for anti-spoofing
        result = await ai_service.detect_anti_spoofing(image_bytes)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anti-spoofing detection API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during anti-spoofing detection"
        )

@router.post("/extract-face-embedding")
async def extract_face_embedding(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Extract face embedding from uploaded image"""
    try:
        logger.info(f"Received file for face embedding: {image.filename}, content_type: {image.content_type}")
        
        # Validate file type - more flexible validation
        if image.content_type and not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be an image, received: {image.content_type}"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Extract face embedding
        result = await ai_service.extract_face_embedding(image_bytes)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face embedding API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during face embedding extraction"
        )
