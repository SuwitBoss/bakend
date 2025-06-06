from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import         # Validate for duplicates using first photo
        # Create a single-image validation request from the first photo
        face_validation_req = FaceValidationRequest(image=user_data.photos[0])
        duplicate_check_response = await validate_face_match(request=face_validation_req, db=db)h2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import base64
import json # Added import
import numpy as np # Added import

from app.core.database import get_db
from app.core.auth import create_access_token, verify_token
from app.core.config import settings
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.user_service import UserService
from app.models.user import User
# Add AIService import
from app.services.ai_service import AIService

router = APIRouter()

# Pydantic models for request/response
class FaceValidationRequest(BaseModel):
    image: str  # base64 encoded image

class FaceConsistencyRequest(BaseModel):
    photos: List[str]  # List of base64 encoded images

class FaceValidationResponse(BaseModel):
    success: bool
    face_detected: bool
    matched_users: List[str] = []
    message: str

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
    ai_service = AIService() # Initialize AIService
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

        # 1. Validate for duplicates
        # Construct FaceValidationRequest for validate_face_match
        face_validation_req = FaceValidationRequest(photos=user_data.photos)
        duplicate_check_response = await validate_face_match(request=face_validation_req, db=db)
        if duplicate_check_response.is_duplicate:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Face already registered to another user: {', '.join(duplicate_check_response.matched_users)}. {duplicate_check_response.message}"
            )

        # 2. Validate face consistency if multiple photos are provided
        if len(photo_data_bytes_list) > 1:
            consistency_result = await ai_service.analyze_face_consistency(photo_data_bytes_list)
            if not consistency_result.get("all_same_person", True):
                # Consider how to handle inconsistent_photos and confidence_scores in the error message
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Faces in provided photos are not consistent. {consistency_result.get('message', '')}"
                )
        
        # 3. Extract AdaFace embedding from the first valid photo for storage
        # (Assuming the first photo is representative and validated for quality if necessary)
        # analyze_registration_photos could be used here if more detailed analysis per photo is needed
        # For now, we'll use the first photo directly after consistency and duplicate checks.
        
        # We need to ensure the photo is suitable (e.g., contains a face).
        # The `extract_face_embedding` method in AIService should handle cases where no face is found.
        first_photo_bytes = photo_data_bytes_list[0]
        embedding_result = await ai_service.extract_face_embedding(first_photo_bytes, model_name="adaface")

        if not embedding_result["success"] or not isinstance(embedding_result.get("embedding"), np.ndarray):
            # Log embedding_result for debugging
            # print(f"Debug: Embedding extraction failed for registration: {embedding_result}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract face embedding for registration. {embedding_result.get('message', 'No embedding found or error in extraction.')}"
            )
        
        face_embedding_np = embedding_result["embedding"]
        # Convert numpy array to list for JSON serialization
        face_embedding_list = face_embedding_np.tolist()
        face_embedding_json = json.dumps(face_embedding_list)

    try:
        # Pass the face_embedding_json to create_user
        user = user_service.create_user(user_data, face_embedding_json=face_embedding_json)
        if not user: # Should not happen if previous checks are fine, but as a safeguard
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user account."
            )
        return UserResponse.from_orm(user)
    except ValueError as e: # Catch errors from user_service.create_user (like duplicate username/email if not pre-checked)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e: # Catch any other unexpected errors during user creation
        # Log the error: print(f"Unexpected error during user creation: {str(e)}")
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

# Add validation schemas
class ValidationResponse(BaseModel):
    available: bool
    message: str

class FaceMatchResponse(BaseModel):
    is_duplicate: bool
    matched_users: List[str] = []
    confidence_scores: List[float] = []
    message: str

# Face Login schemas
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
    
    # Clean the input
    clean_name = full_name.strip() if full_name else ""
    
    # Validation rules
    if not clean_name:
        return {
            "available": False,
            "message": "กรุณากรอกชื่อ-นามสกุล"
        }
    
    if len(clean_name) < 2:
        return {
            "available": False,
            "message": "ชื่อ-นามสกุลต้องมีอย่างน้อย 2 ตัวอักษร"
        }
    
    if len(clean_name) > 100:
        return {
            "available": False,
            "message": "ชื่อ-นามสกุลต้องไม่เกิน 100 ตัวอักษร"
        }
    
    # Check if it's just numbers
    if clean_name.isdigit():
        return {
            "available": False,
            "message": "ชื่อ-นามสกุลไม่สามารถเป็นตัวเลขอย่างเดียวได้"
        }
    
    # Check for invalid characters (allow Thai, English, spaces, dots, hyphens)
    if not re.match(r'^[a-zA-Zก-๙\s\.\-]+$', clean_name):
        return {
            "available": False,
            "message": "ชื่อ-นามสกุลสามารถใช้ได้เฉพาะตัวอักษรไทย อังกฤษ จุด และขีดกลางเท่านั้น"
        }
    
    # Check for excessive spaces or special patterns
    if '  ' in clean_name or clean_name.startswith('.') or clean_name.endswith('.'):
        return {
            "available": False,
            "message": "รูปแบบชื่อ-นามสกุลไม่ถูกต้อง"
        }
    
    # Check if it looks like a real name (at least contains some letters)
    if not re.search(r'[a-zA-Zก-๙]', clean_name):
        return {
            "available": False,
            "message": "ชื่อ-นามสกุลต้องมีตัวอักษร"
        }
    
    return {
        "available": True,
        "message": "ชื่อ-นามสกุลใช้ได้"
    }

@router.post("/validate/face-match", response_model=FaceMatchResponse)
async def validate_face_match(
    request: FaceValidationRequest,
    db: Session = Depends(get_db)
):
    """Check if uploaded faces match existing users in database"""
    try:
        from app.services.ai_service import AIService
        
        ai_service = AIService()
        user_service = UserService(db)
        
        # Decode base64 images
        photo_data_bytes_list = []
        for photo_b64 in request.photos:
            try:
                photo_bytes = base64.b64decode(photo_b64)
                photo_data_bytes_list.append(photo_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image format: {str(e)}"
                )
        
        if not photo_data_bytes_list:
            return FaceMatchResponse(
                is_duplicate=False,
                message="No photos provided for validation."
            )

        # Use the first photo for duplicate checking
        # Assuming analyze_registration_photos or a similar consistency check
        # has already been performed or will be performed.
        new_face_image_bytes = photo_data_bytes_list[0]
        
        embedding_result = await ai_service.extract_face_embedding(new_face_image_bytes, model_name="adaface")
        
        if not embedding_result["success"] or not isinstance(embedding_result.get("embedding"), np.ndarray):
            # Log the actual embedding_result for debugging if needed
            # print(f"Debug: Embedding extraction failed or invalid format: {embedding_result}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract face embedding for validation."
            )
            
        new_embedding = embedding_result["embedding"]
        
        existing_users = user_service.get_all_active_users()
        
        matched_users = []
        confidence_scores = []
        is_duplicate = False
        # Threshold for AdaFace, can be moved to config
        # From docs: AdaFace: Threshold: ~0.4 for loose, ~0.6 for medium, ~0.8 for strict
        similarity_threshold = 0.6 

        for user in existing_users:
            if user.face_embedding:
                try:
                    # Stored embedding is a JSON string of a list
                    stored_embedding_list = json.loads(user.face_embedding)
                    stored_embedding_np = np.array(stored_embedding_list, dtype=np.float32);
                    
                    # Ensure embeddings are 2D for cosine_similarity
                    if new_embedding.ndim == 1:
                        new_embedding_2d = new_embedding.reshape(1, -1)
                    else:
                        new_embedding_2d = new_embedding
                    
                    if stored_embedding_np.ndim == 1:
                        stored_embedding_np_2d = stored_embedding_np.reshape(1, -1)
                    else:
                        stored_embedding_np_2d = stored_embedding_np

                    similarity = ai_service.calculate_similarity(new_embedding_2d, stored_embedding_np_2d)
                    
                    # calculate_similarity returns a 2D array, e.g., [[0.98]]
                    # or a float if inputs were 1D (which we are trying to avoid by reshaping)
                    # For sklearn's cosine_similarity with 2D inputs, result is [[value]]
                    actual_similarity_score = 0.0
                    if isinstance(similarity, np.ndarray) and similarity.ndim == 2:
                        actual_similarity_score = float(similarity[0][0])
                    elif isinstance(similarity, float): # Fallback if it somehow returned a float
                        actual_similarity_score = similarity
                    else:
                        # Log unexpected similarity format
                        # print(f"Debug: Unexpected similarity format: {similarity} for user {user.username}")
                        continue


                    if actual_similarity_score >= similarity_threshold:
                        is_duplicate = True
                        matched_users.append(user.username) # Or user.id, as preferred
                        confidence_scores.append(actual_similarity_score)
                        
                except json.JSONDecodeError:
                    # Log error: print(f"Error decoding JSON for user {user.id}: {user.face_embedding}")
                    continue # Skip user if embedding is malformed
                except Exception as e:
                    # Log other errors: print(f"Error processing user {user.id} embedding: {e}")
                    continue # Skip user on other errors

        if is_duplicate:
            message = f"พบใบหน้าที่ตรงกันกับผู้ใช้: {', '.join(matched_users)}."
        else:
            message = "ไม่พบใบหน้าที่ตรงกับผู้ใช้ที่มีอยู่ในระบบ"
            
        return FaceMatchResponse(
            is_duplicate=is_duplicate,
            matched_users=matched_users,
            confidence_scores=confidence_scores,
            message=message
        )
        
    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        # Log the full error: print(f"Unhandled error in validate_face_match: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face matching error: An unexpected error occurred. {str(e)}"
        )

@router.post("/validate/face-consistency", response_model=dict)
async def validate_face_consistency(
    request: FaceConsistencyRequest,
    db: Session = Depends(get_db)
):
    """Check if all uploaded photos contain the same person"""
    try:
        from app.services.ai_service import AIService
        
        ai_service = AIService()
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
        from app.services.ai_service import AIService
        
        ai_service = AIService()
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
        deepfake_result = await ai_service.detect_deepfake(image_bytes)
        processing_steps["deepfake_detection"] = {
            "completed": True,
            "result": deepfake_result
        }
        
        # Check for deepfake - use is_deepfake field
        if deepfake_result.get("is_deepfake", False):
            return FaceLoginResponse(
                success=False,
                message="Deepfake detected. Authentication failed.",
                processing_steps=processing_steps
            )
        
        # Step 2: Anti-Spoofing Detection
        anti_spoof_result = await ai_service.detect_anti_spoofing(image_bytes)
        processing_steps["anti_spoofing"] = {
            "completed": True,
            "result": anti_spoof_result
        }
        
        if not anti_spoof_result.get("is_live", True):
            return FaceLoginResponse(
                success=False,
                message=f"Spoofing attack detected: {anti_spoof_result.get('attack_type', 'unknown')}",
                processing_steps=processing_steps
            )
        
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
        all_users = user_service.get_all_active_users()
        best_match_user = None
        best_similarity = 0.0
        similarity_threshold = 0.7  # Adjust based on your security requirements
        
        for user in all_users:
            if user.face_embedding:  # Assuming we store face embedding in user model
                try:
                    import json
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

class FaceValidationRequest(BaseModel):
    image: str  # base64 encoded image

class FaceConsistencyRequest(BaseModel):
    photos: List[str]  # List of base64 encoded images

class FaceValidationResponse(BaseModel):
    success: bool
    face_detected: bool
    matched_users: List[str] = []
    message: str

@router.post("/validate/face-match", response_model=FaceValidationResponse)
async def validate_face_match(
    request: FaceValidationRequest,
    db: Session = Depends(get_db)
):
    """ทดสอบการตรวจสอบใบหน้าโดยไม่บันทึกข้อมูล"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        
        # Initialize AI service
        ai_service = AIService()
        
        # Extract face embedding
        embedding_result = await ai_service.extract_face_embedding(image_bytes, model_name="adaface")
        
        if not embedding_result.get("success"):
            return FaceValidationResponse(
                success=False,
                face_detected=False,
                message=embedding_result.get("message", "Failed to extract face embedding")
            )
        
        # Get all users with face embeddings for comparison
        user_service = UserService(db)
        users_with_faces = user_service.get_users_with_face_embeddings()
        
        matched_users = []
        if users_with_faces:
            new_embedding = np.array(embedding_result["embedding"])
            
            for user in users_with_faces:
                if user.face_embedding_json:
                    try:
                        stored_embedding = np.array(json.loads(user.face_embedding_json))
                        similarity = np.dot(new_embedding, stored_embedding) / (
                            np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > 0.6:  # Threshold for match
                            matched_users.append(f"{user.username} (similarity: {similarity:.3f})")
                    except Exception as e:
                        continue
        
        return FaceValidationResponse(
            success=True,
            face_detected=True,
            matched_users=matched_users,
            message=f"Face detected. Found {len(matched_users)} potential matches."
        )
        
    except Exception as e:
        return FaceValidationResponse(
            success=False,
            face_detected=False,
            message=f"Validation error: {str(e)}"
        )
