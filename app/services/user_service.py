from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.auth import get_password_hash, verify_password
from typing import Optional

class UserService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def create_user(self, user_data: UserCreate, face_embedding_json: Optional[str] = None) -> Optional[User]: # Added face_embedding_json
        """Create a new user"""
        try:
            # Check if username or email already exists
            if self.get_user_by_username(user_data.username):
                raise ValueError("Username already exists")
            
            if self.get_user_by_email(user_data.email):
                raise ValueError("Email already exists")
            
            # Create new user
            hashed_password = get_password_hash(user_data.password)
            db_user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                face_embedding=face_embedding_json  # Added this line
            )
            
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            return db_user
            
        except IntegrityError:
            self.db.rollback()
            raise ValueError("Username or email already exists")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
    
    def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user information"""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        update_data = user_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        self.db.commit()
        return True
    
    def update_face_embedding(self, user_id: int, face_embedding: str) -> bool:
        """Update user's face embedding"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False
            
            user.face_embedding = face_embedding
            self.db.commit()
            return True
            
        except Exception:
            self.db.rollback()
            return False
    
    def get_all_active_users(self) -> list[User]:
        """Get all active users"""
        return self.db.query(User).filter(User.is_active == True).all()
    
    def get_user_by_face_embedding(self, face_embedding: str, similarity_threshold: float = 0.8) -> Optional[User]:
        """Find user by face embedding similarity"""
        try:
            # Get all users with face embeddings
            users_with_faces = self.db.query(User).filter(
                User.face_embedding.isnot(None),
                User.is_active == True
            ).all()
            
            if not users_with_faces:
                return None
            
            # Import here to avoid circular imports
            import json
            import numpy as np
            from app.services.ai_service import AIService
            
            ai_service = AIService()
            target_embedding = np.array(json.loads(face_embedding))
            
            best_match = None
            best_similarity = 0.0
            for user in users_with_faces:
                try:
                    stored_embedding = np.array(json.loads(user.face_embedding))
                    similarity = ai_service.calculate_similarity(target_embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = user
                        
                except (json.JSONDecodeError, ValueError):
                    # Skip invalid embeddings
                    continue
            
            return best_match
            
        except Exception:
            return None

    def get_users_with_face_embeddings(self):
        """Get all users that have face embeddings stored"""
        try:
            return self.db.query(User).filter(User.face_embedding.isnot(None)).all()
        except Exception:
            return []
