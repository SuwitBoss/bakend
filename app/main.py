from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os
import onnxruntime as ort
import logging
import time
import psutil
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.api.v1.router import api_router

# Setup logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("facesocial")

# Create FastAPI instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration - Secure for production
origins = [
    "https://facesocial.example.com",  # Specific domains only
    "https://app.facesocial.example.com"
]

# Add development origins if not in production
if not settings.ENVIRONMENT == "production":
    origins.extend([
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://facesocial_frontend:3000"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Only with specific origins
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type", 
        "X-Requested-With"
    ],
    max_age=600,
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Health check endpoint
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "FaceSocial Backend API",
            "version": settings.VERSION,
            "status": "running"
        }
    )

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "facesocial-backend", 
            "message": "Face Social API is running smoothly",
            "timestamp": "2025-06-04"
        }
    )

@app.get("/test")
def simple_test():
    """Simple sync endpoint for testing"""
    return {"message": "Simple test endpoint working", "status": "ok"}

@app.get("/test-async")
async def async_test():
    """Simple async endpoint for testing"""
    return {"message": "Async test endpoint working", "status": "ok"}

# Handle preflight requests
@app.options("/{full_path:path}")
async def options_handler():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# App startup and shutdown events for proper lifecycle management
@app.on_event("startup")
async def startup_event():
    """Initialize resources on app startup"""
    # Log system information
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"⚙️ Environment: {settings.ENVIRONMENT}")
    
    # Check if model directory exists
    models_path = Path(settings.MODELS_PATH)
    if models_path.exists():
        logger.info(f"✅ Model directory found: {models_path}")
        # Log available models
        for model_type in ["face-detection", "face-recognition", "deepfake-detection", "anti-spoofing"]:
            model_dir = models_path / model_type
            if model_dir.exists():
                models = list(model_dir.glob("*.onnx"))
                logger.info(f"📦 Found {len(models)} models in {model_type}: {[m.name for m in models]}")
            else:
                logger.warning(f"⚠️ Model directory not found: {model_dir}")
    else:
        logger.error(f"❌ Model directory not found: {models_path}")
    
    # Log ONNX Runtime information
    logger.info(f"🔧 ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    logger.info(f"🔌 Available ONNX Runtime providers: {providers}")
    
    # Pre-load ONNX DLLs for better performance
    try:
        ort.preload_dlls()
        logger.info("✅ ONNX Runtime DLLs preloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload ONNX Runtime DLLs: {e}")
    
    # Log system resources
    try:
        mem = psutil.virtual_memory()
        logger.info(f"💾 System memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
        if 'CUDAExecutionProvider' in providers:
            logger.info(f"🎮 GPU VRAM limit set to: {settings.VRAM_LIMIT_MB} MB")
    except Exception as e:
        logger.warning(f"⚠️ Error getting system resources: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on app shutdown"""
    logger.info(f"🛑 Shutting down {settings.PROJECT_NAME}")
    
    # Add cleanup code if needed
    
    logger.info("👋 Shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
