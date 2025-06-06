from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
from pydantic import BaseModel
from typing import List
import base64

from app.services.ai_service import AIService
from app.core.config import settings

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

@router.post("/face/detect-realtime")
async def detect_faces_realtime(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Real-time face detection optimized for speed"""
    try:
        logger.info(f"🔍 Real-time detection request received - filename: {image.filename}, content_type: {image.content_type}")
        
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"📁 Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            logger.error("❌ Empty image file received")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
        
        # Detect faces with optimized settings for real-time
        logger.info("🤖 Starting face detection...")
        result = await ai_service.detect_faces_in_image(image_bytes)
        logger.info(f"✅ Detection completed - found {len(result.get('faces', []))} faces")
        
        # Return minimal response for speed
        faces_with_bbox = []
        for face in result.get('faces', []):
            faces_with_bbox.append({
                'bbox': {
                    'x': face['bbox'][0],
                    'y': face['bbox'][1], 
                    'width': face['bbox'][2] - face['bbox'][0],
                    'height': face['bbox'][3] - face['bbox'][1]
                },
                'confidence': face['confidence']
            })
        
        response = {
            "success": True,
            "faces": faces_with_bbox,
            "count": len(faces_with_bbox)
        }
        
        logger.info(f"📤 Returning response: {response['count']} faces detected")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Real-time face detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection error"
        )

class RegistrationPhotosRequest(BaseModel):
    photos: List[str]  # List of base64 encoded images

@router.post("/analyze-registration-photos")
async def analyze_registration_photos(
    request: RegistrationPhotosRequest
) -> Dict[str, Any]:
    """Analyze multiple photos for registration with consistency checks"""
    try:
        if not request.photos:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one photo is required"
            )
        
        if len(request.photos) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 photos allowed"
            )
        
        # Convert base64 images to bytes
        photos_data = []
        for i, photo_b64 in enumerate(request.photos):
            try:
                photo_bytes = base64.b64decode(photo_b64)
                photos_data.append(photo_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 data in photo {i+1}: {str(e)}"
                )
        
        # Analyze photos using AI service
        result = await ai_service.analyze_registration_photos(photos_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration photos analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis error: {str(e)}"
        )
