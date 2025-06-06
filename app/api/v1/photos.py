"""
Photo Analysis API Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
import logging
from app.services.ai_service import AIService
from app.schemas.photo import PhotoAnalysisRequest, PhotoAnalysisResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize AI service
ai_service = AIService()

@router.post("/analyze", response_model=PhotoAnalysisResponse)
async def analyze_photos(
    files: List[UploadFile] = File(..., description="รูปภาพที่ต้องการวิเคราะห์ (5-20 รูป)")
):
    """
    วิเคราะห์รูปภาพสำหรับการลงทะเบียน
    
    Features:
    - ตรวจสอบคุณภาพรูปภาพ
    - วิเคราะห์แสงและความชัดเจน
    - ตรวจจับใบหน้า
    - เปรียบเทียบความเหมือนของใบหน้า
    """
    try:
        # Validate input
        if len(files) < 5:
            raise HTTPException(
                status_code=400, 
                detail="ต้องการรูปภาพอย่างน้อย 5 รูป"
            )
        
        if len(files) > 20:
            raise HTTPException(
                status_code=400, 
                detail="รูปภาพสูงสุด 20 รูป"
            )
        
        # Read photo bytes
        photo_bytes = []
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"ไฟล์ {file.filename} ไม่ใช่รูปภาพ"
                )
            
            # Read file content
            content = await file.read()
            
            # Validate file size (max 10MB)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"ไฟล์ {file.filename} มีขนาดใหญ่เกินไป (สูงสุด 10MB)"
                )
            
            photo_bytes.append(content)
        
        logger.info(f"Analyzing {len(photo_bytes)} photos for registration")
        
        # Perform AI analysis
        analysis_result = await ai_service.analyze_registration_photos(photo_bytes)
        
        # Format response
        response = PhotoAnalysisResponse(
            success=True,
            message="วิเคราะห์รูปภาพเสร็จสิ้น",
            data=analysis_result
        )
        
        logger.info(f"Analysis completed: avg_score={analysis_result.get('average_score', 0):.1f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Photo analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ: {str(e)}"
        )


@router.post("/analyze-single", response_model=Dict[str, Any])
async def analyze_single_photo(
    file: UploadFile = File(..., description="รูปภาพเดี่ยวที่ต้องการวิเคราะห์")
):
    """
    วิเคราะห์รูปภาพเดี่ยว
    
    ใช้สำหรับ real-time analysis ขณะผู้ใช้อัปโหลดรูป
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="ไฟล์ที่อัปโหลดไม่ใช่รูปภาพ"
            )
        
        # Read file
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="ไฟล์มีขนาดใหญ่เกินไป (สูงสุด 10MB)"
            )
          # Quick analysis for single photo
        analysis_result = await ai_service.analyze_single_photo_detailed(content)
        
        return {
            "success": True,
            "message": "วิเคราะห์รูปภาพเดี่ยวเสร็จสิ้น",
            "data": analysis_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single photo analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """ตรวจสอบสถานะของ AI services"""
    try:
        # Test AI service initialization
        status = {
            "ai_service": "ready",
            "face_detection": "ready" if ai_service.face_detector.session else "mock_mode",
            "face_recognition": "ready" if ai_service.face_recognizer.session else "mock_mode",
            "quality_analysis": "ready",
            "models_path": str(ai_service.models_path),
            "status": "healthy"
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI service health check failed: {str(e)}"
        )
