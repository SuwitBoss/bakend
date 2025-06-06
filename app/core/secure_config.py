import os
from typing import Optional

class SecureConfig:
    """Secure configuration management"""
    
    @staticmethod
    def get_database_url() -> str:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        return url
    
    @staticmethod  
    def get_redis_url() -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    @staticmethod
    def get_jwt_secret() -> str:
        secret = os.getenv("JWT_SECRET")
        if not secret:
            raise ValueError("JWT_SECRET environment variable not set")
        return secret
    
    @staticmethod
    def is_production() -> bool:
        return os.getenv("ENVIRONMENT", "development") == "production"
