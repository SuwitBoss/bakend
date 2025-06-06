from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Union
import os
import json

class Settings(BaseSettings):    # Project Info
    PROJECT_NAME: str = "FaceSocial Backend API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered social platform backend services (🔥 Hot Reload Active)"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql://facesocial:password@facesocial_postgres:5432/facesocial"
    
    # Redis
    REDIS_URL: str = "redis://facesocial_redis:6379"
    
    # Security
    SECRET_KEY: str = "docker-secret-key-for-development"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS - Parse JSON string from env or use default list
    ALLOWED_HOSTS: Union[str, List[str]] = Field(default=[
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://facesocial_frontend:3000",
        "*"  # Allow all origins for development
    ])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse ALLOWED_HOSTS if it's a JSON string
        if isinstance(self.ALLOWED_HOSTS, str):
            try:
                self.ALLOWED_HOSTS = json.loads(self.ALLOWED_HOSTS)
            except json.JSONDecodeError:
                # If not valid JSON, treat as single host
                self.ALLOWED_HOSTS = [self.ALLOWED_HOSTS]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "uploads"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # AI Models
    MODELS_PATH: str = "/app/model"
    VRAM_LIMIT_MB: int = 6144  # 6GB
    
    # ONNX Runtime Optimization
    OMP_NUM_THREADS: int = 8  # Default to 8 threads for CPU operations
    OPENBLAS_NUM_THREADS: int = 8  # For matrix operations
    MKL_NUM_THREADS: int = 8  # For Intel MKL optimizations
    MODEL_UNLOAD_TIMEOUT: int = 300  # 5 minutes of inactivity before unloading
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
