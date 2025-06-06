from fastapi import APIRouter

# Import route modules
from .auth import router as auth_router
from .photos import router as photos_router
from .ai import router as ai_router
from .glasses import router as glasses_router

# Create main API router
api_router = APIRouter()

# Test endpoint for Phase 1
@api_router.get("/test")
async def test_endpoint():
    return {"message": "API v1 is working!", "phase": "Phase 3 - AI Integration"}

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(photos_router, prefix="/photos", tags=["photo-analysis"])
api_router.include_router(ai_router, prefix="/ai", tags=["ai-services"])
api_router.include_router(glasses_router, prefix="/glasses", tags=["glasses-detection"])
