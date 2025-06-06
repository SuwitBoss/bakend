"""
Advanced Glasses Detection API Endpoints
Provides multiple detection methods including glasses-detector library
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Dict, Any

from app.services.advanced_glasses_detection import AdvancedGlassesDetector

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the advanced glasses detector
detector = AdvancedGlassesDetector()

@router.post("/detect-glasses-advanced")
async def detect_glasses_advanced(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Advanced glasses detection using multiple algorithms
    """
    try:
        # Read uploaded image
        contents = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform detection
        result = detector.detect_glasses(image)
        
        return JSONResponse(content={
            "status": "success",
            "result": result,
            "message": "Advanced glasses detection completed"
        })
        
    except Exception as e:
        logger.error(f"Error in advanced glasses detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/detect-glasses-base64")
async def detect_glasses_base64(data: Dict[str, str]) -> Dict[str, Any]:
    """
    Advanced glasses detection from base64 encoded image
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        image_data = data["image"]
        if image_data.startswith("data:image"):
            # Remove data URL prefix
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image then to OpenCV
        pil_image = Image.open(BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Perform detection
        result = detector.detect_glasses(image)
        
        return JSONResponse(content={
            "status": "success",
            "result": result,
            "message": "Advanced glasses detection completed"
        })
        
    except Exception as e:
        logger.error(f"Error in base64 glasses detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.get("/detection-methods")
async def get_detection_methods() -> Dict[str, Any]:
    """
    Get information about available detection methods
    """
    try:
        methods = detector.get_available_methods()
        
        return JSONResponse(content={
            "status": "success",
            "methods": methods,
            "message": "Available detection methods retrieved"
        })
        
    except Exception as e:
        logger.error(f"Error getting detection methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get methods: {str(e)}")

@router.post("/benchmark-methods")
async def benchmark_methods(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Benchmark all available detection methods on the same image
    """
    try:
        # Read uploaded image
        contents = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run benchmark
        results = detector.benchmark_methods(image)
        
        return JSONResponse(content={
            "status": "success",
            "benchmark_results": results,
            "message": "Method benchmarking completed"
        })
        
    except Exception as e:
        logger.error(f"Error in method benchmarking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for glasses detection service
    """
    try:
        health_status = detector.health_check()
        
        return JSONResponse(content={
            "status": "success",
            "health": health_status,
            "message": "Health check completed"
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "health": {"overall": "unhealthy", "error": str(e)},
                "message": "Health check failed"
            }
        )
