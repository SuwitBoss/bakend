from fastapi import APIRouter

# Import route modules (will create these)
# from .auth import router as auth_router
# from .users import router as users_router

# Create main API router
api_router = APIRouter()

# Test endpoint for Phase 1
@api_router.get("/test")
async def test_endpoint():
    return {"message": "API is working!", "phase": "Phase 1 - Foundation"}

# Include routers (uncomment as we create them)
# api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
# api_router.include_router(users_router, prefix="/users", tags=["users"])
