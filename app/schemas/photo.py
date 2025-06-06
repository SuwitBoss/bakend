"""
Photo Analysis Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PhotoAnalysisRequest(BaseModel):
    """Request model for photo analysis"""
    photos: List[str] = Field(..., description="Base64 encoded photos")

class FaceDetection(BaseModel):
    """Face detection result"""
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence")
    landmarks: Optional[List[List[float]]] = Field(None, description="Facial landmarks")

class SimilarityAnalysis(BaseModel):
    """Face similarity analysis result"""
    index: int = Field(..., description="Photo index")
    similarity: float = Field(..., description="Similarity score (0-1)")
    is_same_person: bool = Field(..., description="Whether it's the same person")
    confidence: float = Field(..., description="Confidence score")

class PhotoAnalysisData(BaseModel):
    """Photo analysis result data"""
    total_photos: int = Field(..., description="Total number of photos analyzed")
    quality_scores: List[float] = Field(..., description="Quality scores (0-100)")
    lighting_scores: List[float] = Field(..., description="Lighting scores (0-100)")
    face_detections: List[bool] = Field(..., description="Face detection results")
    similarity_analysis: List[SimilarityAnalysis] = Field(..., description="Face similarity analysis")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    passed_quality: int = Field(..., description="Number of photos that passed quality check")
    average_score: float = Field(..., description="Average quality score")

class PhotoAnalysisResponse(BaseModel):
    """Response model for photo analysis"""
    success: bool = Field(..., description="Whether analysis was successful")
    message: str = Field(..., description="Response message")
    data: PhotoAnalysisData = Field(..., description="Analysis results")

class SinglePhotoAnalysisData(BaseModel):
    """Single photo analysis result"""
    quality_score: float = Field(..., description="Quality score (0-100)")
    lighting_score: float = Field(..., description="Lighting score (0-100)")
    face_detected: bool = Field(..., description="Whether face was detected")
    recommendations: List[str] = Field(..., description="Improvement recommendations")

class SinglePhotoAnalysisResponse(BaseModel):
    """Response model for single photo analysis"""
    success: bool = Field(..., description="Whether analysis was successful")
    message: str = Field(..., description="Response message")
    data: SinglePhotoAnalysisData = Field(..., description="Analysis results")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    error_code: Optional[str] = Field(None, description="Error code")
