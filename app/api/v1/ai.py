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

# Request models for new endpoints
class SentimentRequest(BaseModel):
    text: str

class HashtagRequest(BaseModel):
    text: str
    max_hashtags: int = 10

@router.post("/generate-caption")
async def generate_caption(
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Generate captions for uploaded images using AI"""
    try:
        logger.info(f"Received file for caption generation: {image.filename}, content_type: {image.content_type}")
        
        # Validate file type
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
        
        # Generate captions
        result = await ai_service.generate_caption(image_bytes)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Caption generation API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during caption generation"
        )

@router.post("/analyze-sentiment")
async def analyze_sentiment(
    request: SentimentRequest
) -> Dict[str, Any]:
    """Analyze sentiment of text content"""
    try:
        logger.info(f"Received text for sentiment analysis (length: {len(request.text)})")
        
        # Validate text
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content is required"
            )
        
        if len(request.text) > 2000:  # Reasonable limit for social media posts
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content too long (maximum 2000 characters)"
            )
        
        # Analyze sentiment
        result = await ai_service.analyze_sentiment(request.text)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during sentiment analysis"
        )

@router.post("/suggest-hashtags")
async def suggest_hashtags(
    request: HashtagRequest
) -> Dict[str, Any]:
    """Suggest relevant hashtags for posts based on content"""
    try:
        logger.info(f"Received text for hashtag suggestions (length: {len(request.text)}, max_hashtags: {request.max_hashtags})")
        
        # Validate text
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content is required"
            )
        
        if len(request.text) > 1000:  # Reasonable limit for hashtag analysis
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content too long (maximum 1000 characters)"
            )
        
        # Validate max_hashtags
        if request.max_hashtags < 1 or request.max_hashtags > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_hashtags must be between 1 and 50"
            )
        
        # Suggest hashtags
        result = await ai_service.suggest_hashtags(request.text, request.max_hashtags)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hashtag suggestion API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during hashtag suggestion"
        )
