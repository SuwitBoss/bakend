from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.core.secure_config import SecureConfig

# Optimized database engine configuration
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30, 
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "query_cache_size": 1200,
    "echo": False
}

engine = create_async_engine(
    SecureConfig.get_database_url(),
    **DATABASE_CONFIG
)

AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# Dependency for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
