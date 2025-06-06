import numpy as np
from typing import List, Dict, Any
import asyncio
import cv2
from fastapi import HTTPException
import logging
from .antispoofing_service import AntispoofingService

class AIService:
    """AI Service for face detection and related operations"""
    
    def __init__(self):
        """Initialize AI service with required models and configurations"""
        self.face_cascade = None
        self.logger = logging.getLogger(__name__)
        self.antispoofing_service = AntispoofingService()
        
    async def initialize(self):
        """Initialize async components and load models"""
        try:
            # Load face detection models
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'            )
            self.logger.info("AI Service initialized successfully")
            
            # Initialize antispoofing service
            await self.antispoofing_service.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}")
            raise
    
    async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the provided image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with coordinates and confidence
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image provided")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Format results
            detected_faces = []
            for (x, y, w, h) in faces:
                face_data = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": 0.85  # Placeholder confidence score
                }
                detected_faces.append(face_data)
            
            self.logger.info(f"Detected {len(detected_faces)} faces")
            return detected_faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face detection error: {str(e)}")
        
    async def check_face_spoofing(self, image: np.ndarray, face_bbox: Dict[str, int]) -> Dict[str, Any]:
        """
        Check if a detected face is real or spoofed
        
        Args:
            image: Input image as numpy array
            face_bbox: Face bounding box with x, y, width, height
            
        Returns:
            Anti-spoofing results
        """
        try:
            # Call antispoofing service
            result = await self.antispoofing_service.detect_spoofing(image, face_bbox)
            return result
        except Exception as e:
            self.logger.error(f"Face anti-spoofing check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face anti-spoofing error: {str(e)}")
