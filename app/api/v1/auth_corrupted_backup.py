from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import base64
import json
import numpy as np
import logging

from app.core.database import get_db
from app.core.auth import create_access_token, verify_token
from app.core.config import settings
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.user_service import UserService
from app.models.user import User
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class FaceValidationRequest(BaseModel):
    image: str  # base64 encoded image

class FaceMultipleValidationRequest(BaseModel):
    photos: List[str]  # List of base64 encoded images for duplicate checking

class FaceConsistencyRequest(BaseModel):
    photos: List[str]  # List of base64 encoded images

class FaceValidationResponse(BaseModel):
    success: bool
    face_detected: bool
    matched_users: List[str] = []
    message: str

class ValidationResponse(BaseModel):
    available: bool
    message: str

class FaceMatchResponse(BaseModel):
    is_duplicate: bool
    matched_users: List[str] = []
    confidence_scores: List[float] = []
    message: str

class FaceLoginRequest(BaseModel):
    image: str  # Base64 encoded image
    
class FaceLoginResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_steps: Optional[dict] = None

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> UserResponse:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = verify_token(token)
    if username is None:
        raise credentials_exception
    
    user_service = UserService(db)
    user = user_service.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    return UserResponse.from_orm(user)

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user, optionally with face enrollment"""
    user_service = UserService(db)
    ai_service = AIService(models_path=settings.MODELS_PATH)
    face_embedding_json: Optional[str] = None

    # Validate username and email availability first
    if user_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    if user_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    # Enforce face registration - photos must be provided
    if not user_data.photos or len(user_data.photos) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Face photos are required for registration"
        )

    if user_data.photos and len(user_data.photos) > 0:
        photo_data_bytes_list = []
        for photo_b64 in user_data.photos:
            try:
                photo_bytes = base64.b64decode(photo_b64)
                photo_data_bytes_list.append(photo_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image format in registration photos: {str(e)}"
                )

        if not photo_data_bytes_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid photos provided for face registration."
            )

        # 1. Validate for duplicates using first photo
        face_validation_req = FaceValidationRequest(image=user_data.photos[0])
        duplicate_check_response = await validate_face_match(request=face_validation_req, db=db)
        if duplicate_check_response.matched_users:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Face already registered to another user: {', '.join(duplicate_check_response.matched_users)}. {duplicate_check_response.message}"
            )

        # 2. Validate face consistency if multiple photos are provided
        if len(photo_data_bytes_list) > 1:
            consistency_result = await ai_service.analyze_face_consistency(photo_data_bytes_list)
            if not consistency_result.get("all_same_person", True):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Faces in provided photos are not consistent. {consistency_result.get('message', '')}"
                )
          # 3. Extract AdaFace embedding from the first valid photo for storage
        first_photo_bytes = photo_data_bytes_list[0]
        embedding_result = await ai_service.extract_face_embedding(first_photo_bytes, model_name="adaface")

        if not embedding_result["success"] or not embedding_result.get("embedding"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract face embedding for registration. {embedding_result.get('message', 'No embedding found or error in extraction.')}"
            )
        
        # The embedding is already a list from the AI service
        face_embedding_list = embedding_result["embedding"]
        face_embedding_json = json.dumps(face_embedding_list)

    try:
        user = user_service.create_user(user_data, face_embedding_json=face_embedding_json)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user account."
            )
        return UserResponse.from_orm(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during registration: {str(e)}"
        )

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    user_service = UserService(db)
    user = user_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.get("/test-protected")
async def test_protected_route(current_user: UserResponse = Depends(get_current_user)):
    """Test protected route"""
    return {
        "message": f"Hello {current_user.username}! This is a protected route.",
        "user_id": current_user.id,
        "phase": "Phase 2 - Authentication"
    }

@router.get("/validate/username")
async def validate_username(
    username: str = Query(..., description="Username to validate"),
    db: Session = Depends(get_db)
):
    """Validate if username is available"""
    user_service = UserService(db)
    existing_user = user_service.get_user_by_username(username)
    
    return {
        "available": existing_user is None,
        "message": "Username is available" if existing_user is None else "Username already exists"
    }

@router.get("/validate/email")
async def validate_email(
    email: str = Query(..., description="Email to validate"),
    db: Session = Depends(get_db)
):
    """Validate if email is available"""
    user_service = UserService(db)
    existing_user = user_service.get_user_by_email(email)
    
    return {
        "available": existing_user is None,
        "message": "Email is available" if existing_user is None else "Email already exists"
    }

@router.get("/validate/full_name")
async def validate_full_name(
    full_name: str = Query(..., description="Full name to validate")
):
    """Validate full name format"""
    import re
    
    clean_name = full_name.strip() if full_name else ""
    
    if not clean_name:
        return {"available": False, "message": "กรุณากรอกชื่อ-นามสกุล"}
    
    if len(clean_name) < 2:
        return {"available": False, "message": "ชื่อ-นามสกุลต้องมีอย่างน้อย 2 ตัวอักษร"}
    
    if len(clean_name) > 100:
        return {"available": False, "message": "ชื่อ-นามสกุลต้องไม่เกิน 100 ตัวอักษร"}
    
    if clean_name.isdigit():
        return {"available": False, "message": "ชื่อ-นามสกุลไม่สามารถเป็นตัวเลขอย่างเดียวได้"}
    
    if not re.match(r'^[a-zA-Zก-๙\s\.\-]+$', clean_name):
        return {"available": False, "message": "ชื่อ-นามสกุลสามารถใช้ได้เฉพาะตัวอักษรไทย อังกฤษ จุด และขีดกลางเท่านั้น"}
    
    if '  ' in clean_name or clean_name.startswith('.') or clean_name.endswith('.'):
        return {"available": False, "message": "รูปแบบชื่อ-นามสกุลไม่ถูกต้อง"}
    
    if not re.search(r'[a-zA-Zก-๙]', clean_name):
        return {"available": False, "message": "ชื่อ-นามสกุลต้องมีตัวอักษร"}
    
    return {"available": True, "message": "ชื่อ-นามสกุลใช้ได้"}

@router.post("/validate/face-match", response_model=FaceMatchResponse)
async def validate_face_match(
    request: FaceValidationRequest,
    db: Session = Depends(get_db)
):
    """ทดสอบการตรวจสอบใบหน้าโดยไม่บันทึกข้อมูล - รับรูปเดียว"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        
        # Initialize AI service
        ai_service = AIService(models_path=settings.MODELS_PATH)
        
        # Extract face embedding
        embedding_result = await ai_service.extract_face_embedding(image_bytes, model_name="adaface")
        
        if not embedding_result.get("success"):
            return FaceMatchResponse(
                is_duplicate=False,
                message=embedding_result.get("message", "Failed to extract face embedding")
            )
          # Get all users with face embeddings for comparison
        user_service = UserService(db)
        users_with_faces = user_service.get_users_with_face_embeddings()
        
        matched_users = []
        confidence_scores = []
        is_duplicate = False
        
        if users_with_faces:
            new_embedding = np.array(embedding_result["embedding"])
            
            for user in users_with_faces:
                if user.face_embedding:
                    try:
                        stored_embedding = np.array(json.loads(user.face_embedding))
                        similarity = np.dot(new_embedding, stored_embedding) / (
                            np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > 0.2:  # Threshold for match
                            is_duplicate = True
                            matched_users.append(user.username)
                            confidence_scores.append(float(similarity))
                    except Exception as e:
                        continue
          # Generate appropriate message based on matches found
        if is_duplicate:
            message = f"❌ พบใบหน้าซ้ำกันในระบบ! ตรงกับผู้ใช้: {', '.join(matched_users)}"
            if len(matched_users) == 1:
                message += "\n💡 กรุณาใช้รูปใบหน้าอื่นที่ไม่มีในระบบแล้ว"
            else:
                message += f"\n💡 พบความคล้ายคลึงกับ {len(matched_users)} บัญชีผู้ใช้"
        else:
            message = "✅ ไม่พบใบหน้าซ้ำกันในระบบ สามารถใช้รูปนี้ได้"
            
        return FaceMatchResponse(
            is_duplicate=is_duplicate,
            matched_users=matched_users,
            confidence_scores=confidence_scores,
            message=message
        )
        
    except Exception as e:
        return FaceMatchResponse(
            is_duplicate=False,
            message=f"Validation error: {str(e)}"
        )

@router.post("/validate/face-match-multiple", response_model=FaceMatchResponse)
async def validate_face_match_multiple(
    request: FaceMultipleValidationRequest,
    db: Session = Depends(get_db)
):
    """ทดสอบการตรวจสอบใบหน้าซ้ำกันในฐานข้อมูล - รับหลายรูป"""
    try:
        if not request.photos:
            return FaceMatchResponse(
                is_duplicate=False,
                message="ไม่มีรูปภาพที่ส่งมา"
            )
        
        # Use first photo for duplicate checking
        image_bytes = base64.b64decode(request.photos[0])
        
        # Initialize AI service
        ai_service = AIService(models_path=settings.MODELS_PATH)
        
        # Extract face embedding
        embedding_result = await ai_service.extract_face_embedding(image_bytes, model_name="adaface")
        
        if not embedding_result.get("success"):
            return FaceMatchResponse(
                is_duplicate=False,
                message=embedding_result.get("message", "ไม่สามารถแยกใบหน้าได้")
            )
          # Get all users with face embeddings for comparison
        user_service = UserService(db)
        users_with_faces = user_service.get_users_with_face_embeddings()
        
        matched_users = []
        confidence_scores = []
        is_duplicate = False
        
        if users_with_faces:
            new_embedding = np.array(embedding_result["embedding"])
            
            for user in users_with_faces:
                if user.face_embedding:
                    try:
                        stored_embedding = np.array(json.loads(user.face_embedding))
                        similarity = np.dot(new_embedding, stored_embedding) / (
                            np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > 0.2:  # Threshold for match
                            is_duplicate = True
                            matched_users.append(user.username)
                            confidence_scores.append(float(similarity))
                    except Exception as e:
                        continue
          # Generate appropriate message based on matches found
        if is_duplicate:
            message = f"❌ พบใบหน้าซ้ำกันในระบบ! ตรงกับผู้ใช้: {', '.join(matched_users)}"
            if len(matched_users) == 1:
                message += "\n💡 กรุณาใช้รูปใบหน้าอื่นที่ไม่มีในระบบแล้ว"
            else:
                message += f"\n💡 พบความคล้ายคลึงกับ {len(matched_users)} บัญชีผู้ใช้"
        else:
            message = "✅ ไม่พบใบหน้าซ้ำกันในระบบ สามารถใช้รูปนี้ได้"
            
        return FaceMatchResponse(
            is_duplicate=is_duplicate,
            matched_users=matched_users,
            confidence_scores=confidence_scores,
            message=message
        )
        
    except Exception as e:
        return FaceMatchResponse(
            is_duplicate=False,
            message=f"เกิดข้อผิดพลาดในการตรวจสอบ: {str(e)}"
        )

@router.post("/validate/face-consistency", response_model=dict)
async def validate_face_consistency(
    request: FaceConsistencyRequest,
    db: Session = Depends(get_db)
):
    """Check if all uploaded photos contain the same person"""
    try:
        ai_service = AIService(models_path=settings.MODELS_PATH)
        
        # Decode base64 images
        photo_data = []
        for photo_b64 in request.photos:
            try:
                photo_bytes = base64.b64decode(photo_b64)
                photo_data.append(photo_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image format: {str(e)}"
                )
        
        if len(photo_data) < 2:
            return {
                "all_same_person": True,
                "inconsistent_photos": [],
                "confidence_scores": [],
                "message": "ต้องมีรูปภาพอย่างน้อย 2 รูปในการตรวจสอบ"
            }
        
        # Analyze face consistency using AI service
        result = await ai_service.analyze_face_consistency(photo_data)
        
        return {
            "all_same_person": result.get("all_same_person", True),
            "inconsistent_photos": result.get("inconsistent_photos", []),
            "confidence_scores": result.get("confidence_scores", []),
            "message": result.get("message", "การตรวจสอบเสร็จสิ้น")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face consistency check error: {str(e)}"
        )

@router.post("/login/face", response_model=FaceLoginResponse)
async def login_with_face(
    request: FaceLoginRequest,
    db: Session = Depends(get_db)
):
    """Login using face recognition with deepfake detection, anti-spoofing, and face recognition"""
    try:
        ai_service = AIService(models_path=settings.MODELS_PATH)
        user_service = UserService(db)
        
        processing_steps = {
            "deepfake_detection": {"completed": False, "result": None},
            "anti_spoofing": {"completed": False, "result": None},
            "face_detection": {"completed": False, "result": None},
            "face_recognition": {"completed": False, "result": None}
        }
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image)
        except Exception as e:
            return FaceLoginResponse(
                success=False,
                message=f"Invalid image format: {str(e)}",
                processing_steps=processing_steps
            )
          # Step 1: Deepfake Detection
        try:
            deepfake_result = await ai_service.detect_deepfake(image_bytes)
            processing_steps["deepfake_detection"] = {
                "completed": True,
                "result": deepfake_result
            }
            
            if deepfake_result.get("is_deepfake", False):
                return FaceLoginResponse(
                    success=False,
                    message="Deepfake detected. Authentication rejected.",
                    processing_steps=processing_steps
                )
        except Exception as e:
            processing_steps["deepfake_detection"] = {
                "completed": True,
                "result": {"is_deepfake": False, "confidence": 0.0, "message": f"Error in deepfake detection: {str(e)}"}
            }
            logger.warning(f"Deepfake detection error: {e}, continuing with authentication")
        
        # Step 2: Anti-Spoofing Detection
        try:
            anti_spoofing_result = await ai_service.detect_anti_spoofing(image_bytes)
            processing_steps["anti_spoofing"] = {
                "completed": True,
                "result": anti_spoofing_result
            }
            
            if not anti_spoofing_result.get("is_live", True):
                return FaceLoginResponse(
                    success=False,
                    message="Presentation attack detected. Authentication rejected.",
                    processing_steps=processing_steps
                )
        except Exception as e:
            processing_steps["anti_spoofing"] = {
                "completed": True,
                "result": {"is_live": True, "confidence": 0.0, "message": f"Error in anti-spoofing detection: {str(e)}"}
            }
            logger.warning(f"Anti-spoofing detection error: {e}, continuing with authentication")
        
        # Step 3: Face Detection
        face_detection_result = await ai_service.detect_faces_in_image(image_bytes)
        processing_steps["face_detection"] = {
            "completed": True,
            "result": face_detection_result
        }
        
        if not face_detection_result.get("faces_detected", False):
            return FaceLoginResponse(
                success=False,
                message="No face detected in the image",
                processing_steps=processing_steps
            )
        
        # Step 4: Face Recognition - Extract embedding and match with stored embeddings
        face_recognition_result = await ai_service.extract_face_embedding(image_bytes)
        processing_steps["face_recognition"] = {
            "completed": True,
            "result": face_recognition_result
        }
        
        if not face_recognition_result.get("embedding"):
            return FaceLoginResponse(
                success=False,
                message="Failed to extract face embedding",
                processing_steps=processing_steps
            )
        
        current_embedding = face_recognition_result["embedding"]
        
        # Get all users and their face embeddings for comparison
        all_users = user_service.get_users_with_face_embeddings()
        best_match_user = None
        best_similarity = 0.0
        similarity_threshold = 0.7  # Adjust based on your security requirements
        
        for user in all_users:
            if user.face_embedding:
                try:
                    # Parse the JSON string to get the actual embedding array
                    stored_embedding = json.loads(user.face_embedding)
                    similarity = ai_service.calculate_similarity(current_embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match_user = user
                except Exception as e:
                    continue  # Skip this user if embedding comparison fails
        
        if best_match_user:
            # Generate access token for the matched user
            access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                subject=best_match_user.username, 
                expires_delta=access_token_expires
            )
            
            return FaceLoginResponse(
                success=True,
                message=f"Login successful! Welcome, {best_match_user.full_name}",
                user_id=best_match_user.id,
                username=best_match_user.username,
                access_token=access_token,
                token_type="bearer",
                confidence_score=best_similarity,
                processing_steps=processing_steps
            )
        else:
            return FaceLoginResponse(
                success=False,
                message="Face not recognized. Please try again or use password login.",
                confidence_score=best_similarity,
                processing_steps=processing_steps
            )
            
    except Exception as e:
        return FaceLoginResponse(
            success=False,
            message=f"Face login error: {str(e)}",
            processing_steps=processing_steps
        )
