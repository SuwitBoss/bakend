import numpy as np
from typing import List, Dict, Any
import asyncio
import cv2
from fastapi import HTTPException
import logging
from .antispoofing_service import AntispoofingService
from .deepfake_service import DeepfakeDetectionService
import io
from PIL import Image
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import re
from pathlib import Path

class AIService:
    """AI Service for face detection and related operations"""
    
    def __init__(self):
        """Initialize AI service with required models and configurations"""
        self.face_cascade = None
        self.logger = logging.getLogger(__name__)
        self.antispoofing_service = AntispoofingService()
        
        # Initialize deepfake detection service
        models_path = Path("/app/model")
        self.deepfake_service = DeepfakeDetectionService(models_path)
          # Initialize new AI models (will be loaded lazily)
        self.caption_processor = None
        self.caption_model = None
        self.sentiment_analyzer = None
        self.hashtag_keywords = None
        
    async def initialize(self):
        """Initialize async components and load models"""
        try:
            # Load face detection models
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.logger.info("AI Service initialized successfully")
            
            # Initialize antispoofing service
            await self.antispoofing_service.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Service: {e}")
            raise
    
    def _bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            img_array = np.array(image)
            # Convert RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        except Exception as e:
            self.logger.error(f"Failed to convert bytes to image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
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
    
    async def detect_faces_in_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect faces in image from bytes data
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with detected faces and metadata
        """
        try:
            # Convert bytes to image array
            image = self._bytes_to_image(image_bytes)
              # Use existing detect_faces method
            faces = await self.detect_faces(image)
            
            return {
                "faces": faces,
                "count": len(faces),
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Face detection in image failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face detection error: {str(e)}")
    
    async def detect_deepfake(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect if image contains deepfake content
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Deepfake detection results
        """
        try:
            self.logger.info("Performing deepfake detection using ONNX model...")
            
            # Use the deepfake detection service
            result = await self.deepfake_service.detect_deepfake_from_bytes(image_bytes)
            
            # Detect faces for additional info
            image = self._bytes_to_image(image_bytes)
            faces = await self.detect_faces(image)
            
            # Add faces count to result
            result["faces_detected"] = len(faces)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deepfake detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Deepfake detection error: {str(e)}")
    
    async def detect_anti_spoofing(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect anti-spoofing for image
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Anti-spoofing detection results
        """
        try:
            # Convert bytes to image array
            image = self._bytes_to_image(image_bytes)
            
            # First detect faces
            faces = await self.detect_faces(image)
            
            if not faces:
                return {
                    "is_real": False,
                    "confidence": 0.0,
                    "reason": "No faces detected in image",
                    "faces_detected": 0
                }
            
            # Use the first detected face for anti-spoofing check
            face = faces[0]
            face_bbox = {
                "x": face["x"],
                "y": face["y"],
                "width": face["width"],
                "height": face["height"]
            }
            
            # Use existing anti-spoofing method
            result = await self.check_face_spoofing(image, face_bbox)
            
            return {
                "is_real": result.get("is_real", False),
                "confidence": result.get("confidence", 0.0),
                "faces_detected": len(faces),
                "analysis": result
            }
            
        except Exception as e:
            self.logger.error(f"Anti-spoofing detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Anti-spoofing detection error: {str(e)}")
    
    async def extract_face_embedding(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract face embedding from image
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Face embedding results
        """
        try:
            # Convert bytes to image array
            image = self._bytes_to_image(image_bytes)
            
            # First detect faces
            faces = await self.detect_faces(image)
            
            if not faces:
                return {
                    "success": False,
                    "error": "No faces detected in image",
                    "faces_detected": 0
                }
            
            # For now, implement a placeholder face embedding extraction
            # In a real implementation, this would use a trained face recognition model
            self.logger.info("Extracting face embedding...")
            
            # Use the first detected face
            face = faces[0]
            
            # Placeholder embedding (128-dimensional vector)
            embedding = np.random.rand(128).tolist()
            
            return {
                "success": True,
                "embedding": embedding,
                "embedding_size": len(embedding),
                "face_bbox": {
                    "x": face["x"],
                    "y": face["y"],
                    "width": face["width"],
                    "height": face["height"]
                },
                "confidence": face["confidence"],
                "faces_detected": len(faces)
            }
            
        except Exception as e:
            self.logger.error(f"Face embedding extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face embedding extraction error: {str(e)}")
    
    async def analyze_registration_photos(self, photos_data: List[bytes]) -> Dict[str, Any]:
        """
        Analyze multiple photos for registration with consistency checks
        
        Args:
            photos_data: List of image data as bytes
            
        Returns:
            Registration analysis results
        """
        try:
            self.logger.info(f"Analyzing {len(photos_data)} registration photos...")
            
            results = []
            embeddings = []
            
            for i, photo_bytes in enumerate(photos_data):
                try:
                    # Detect faces in each photo
                    face_result = await self.detect_faces_in_image(photo_bytes)
                    
                    # Extract embedding
                    embedding_result = await self.extract_face_embedding(photo_bytes)
                    
                    # Anti-spoofing check
                    antispoofing_result = await self.detect_anti_spoofing(photo_bytes)
                    
                    photo_analysis = {
                        "photo_index": i,
                        "faces_detected": face_result["count"],
                        "has_valid_face": face_result["count"] > 0,
                        "is_real": antispoofing_result.get("is_real", False),
                        "antispoofing_confidence": antispoofing_result.get("confidence", 0.0),
                        "embedding_extracted": embedding_result.get("success", False)
                    }
                    
                    if embedding_result.get("success"):
                        embeddings.append(embedding_result["embedding"])
                    
                    results.append(photo_analysis)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing photo {i}: {e}")
                    results.append({
                        "photo_index": i,
                        "error": str(e),
                        "has_valid_face": False,
                        "is_real": False
                    })
            
            # Calculate consistency metrics
            valid_photos = [r for r in results if r.get("has_valid_face", False)]
            real_photos = [r for r in valid_photos if r.get("is_real", False)]
            
            # Calculate embedding similarity (placeholder)
            embedding_consistency = 0.8 if len(embeddings) > 1 else 1.0
            
            return {
                "success": True,
                "total_photos": len(photos_data),
                "valid_photos": len(valid_photos),
                "real_photos": len(real_photos),
                "consistency_score": embedding_consistency,
                "recommendation": "approved" if len(real_photos) >= 3 else "rejected",
                "photos_analysis": results,
                "summary": {
                    "quality_score": len(real_photos) / len(photos_data) if photos_data else 0,
                    "consistency_score": embedding_consistency,
                    "overall_score": (len(real_photos) / len(photos_data) + embedding_consistency) / 2 if photos_data else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Registration photos analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Registration analysis error: {str(e)}")
    
    async def _load_caption_model(self):
        """Lazy load caption generation model"""
        if self.caption_processor is None or self.caption_model is None:
            try:
                self.logger.info("Loading caption generation model...")
                # Use BLIP model for image captioning
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.logger.info("Caption generation model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load caption model: {e}")
                raise
    
    async def _load_sentiment_analyzer(self):
        """Lazy load sentiment analysis model"""
        if self.sentiment_analyzer is None:
            try:
                self.logger.info("Loading sentiment analysis model...")
                # Use transformers pipeline for sentiment analysis
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                                 return_all_scores=True)
                self.logger.info("Sentiment analysis model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load sentiment analyzer: {e}")
                raise
    
    async def _load_hashtag_keywords(self):
        """Initialize hashtag keyword database"""
        if self.hashtag_keywords is None:
            # Predefined hashtags by category for social media
            self.hashtag_keywords = {
                'emotions': ['happy', 'sad', 'excited', 'love', 'fun', 'amazing', 'beautiful', 'awesome', 'great'],
                'activities': ['travel', 'food', 'workout', 'party', 'work', 'study', 'cooking', 'reading', 'shopping'],
                'nature': ['sunset', 'beach', 'mountain', 'forest', 'ocean', 'sky', 'flowers', 'trees'],
                'social': ['friends', 'family', 'together', 'group', 'celebration', 'meeting'],
                'lifestyle': ['fashion', 'style', 'home', 'coffee', 'breakfast', 'dinner', 'weekend'],
                'technology': ['phone', 'computer', 'app', 'digital', 'online', 'social media'],
                'entertainment': ['music', 'movie', 'game', 'concert', 'show', 'performance'],
                'general': ['life', 'day', 'time', 'moment', 'experience', 'memory', 'photo']
            }
    
    async def generate_caption(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Generate captions for uploaded images using AI
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Generated captions with confidence scores
        """
        try:
            await self._load_caption_model()
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate caption
            inputs = self.caption_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                output = self.caption_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
            
            caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
            
            # Generate multiple variations
            captions = []
            for i in range(3):  # Generate 3 variations
                with torch.no_grad():
                    output = self.caption_model.generate(**inputs, max_length=50, num_beams=5, 
                                                       do_sample=True, temperature=0.7)
                variant = self.caption_processor.decode(output[0], skip_special_tokens=True)
                captions.append({
                    "text": variant,
                    "confidence": 0.85 - (i * 0.05)  # Simulated confidence scores
                })
            
            self.logger.info(f"Generated {len(captions)} captions")
            
            return {
                "success": True,
                "primary_caption": caption,
                "alternative_captions": captions,
                "total_captions": len(captions) + 1
            }
            
        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Caption generation error: {str(e)}")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text content
        
        Args:
            text: Text content to analyze
            
        Returns:
            Sentiment analysis results with scores
        """
        try:
            await self._load_sentiment_analyzer()
            
            if not text or len(text.strip()) == 0:
                raise ValueError("Text content is required for sentiment analysis")
            
            # Clean text
            cleaned_text = text.strip()
            if len(cleaned_text) > 512:  # Truncate for model limits
                cleaned_text = cleaned_text[:512]
            
            # Analyze sentiment
            results = self.sentiment_analyzer(cleaned_text)
            
            # Process results
            sentiment_scores = {}
            primary_sentiment = None
            max_score = 0
            
            for result in results[0]:  # results is a list with one element containing all scores
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
                
                if score > max_score:
                    max_score = score
                    primary_sentiment = label
            
            # Map labels to common sentiment names
            label_mapping = {
                'label_0': 'negative',
                'label_1': 'neutral', 
                'label_2': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'positive': 'positive'
            }
            
            # Normalize sentiment scores
            normalized_scores = {}
            for label, score in sentiment_scores.items():
                normalized_label = label_mapping.get(label, label)
                normalized_scores[normalized_label] = round(score, 3)
            
            primary_normalized = label_mapping.get(primary_sentiment, primary_sentiment)
            
            self.logger.info(f"Sentiment analysis completed: {primary_normalized} ({max_score:.3f})")
            
            return {
                "success": True,
                "primary_sentiment": primary_normalized,
                "confidence": round(max_score, 3),
                "all_scores": normalized_scores,
                "text_length": len(text),
                "processed_text_length": len(cleaned_text)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")
    
    async def suggest_hashtags(self, text: str, max_hashtags: int = 10) -> Dict[str, Any]:
        """
        Suggest relevant hashtags for posts based on content
        
        Args:
            text: Post content to analyze
            max_hashtags: Maximum number of hashtags to suggest
            
        Returns:
            Suggested hashtags with relevance scores
        """
        try:
            await self._load_hashtag_keywords()
            
            if not text or len(text.strip()) == 0:
                raise ValueError("Text content is required for hashtag suggestions")
            
            # Clean and normalize text
            cleaned_text = text.lower().strip()
            words = re.findall(r'\b\w+\b', cleaned_text)
            
            # Find matching hashtags
            suggested_hashtags = []
            word_set = set(words)
            
            # Check each category for keyword matches
            for category, keywords in self.hashtag_keywords.items():
                for keyword in keywords:
                    if keyword in word_set or keyword in cleaned_text:
                        # Calculate relevance score based on frequency and position
                        frequency = cleaned_text.count(keyword)
                        relevance = min(frequency * 0.3 + 0.7, 1.0)  # Base score + frequency bonus
                        
                        suggested_hashtags.append({
                            "hashtag": f"#{keyword}",
                            "category": category,
                            "relevance": round(relevance, 3),
                            "keyword": keyword
                        })            # Add some trending/popular hashtags based on sentiment
            sentiment_result = await self.analyze_sentiment(text)
            sentiment = sentiment_result.get("primary_sentiment", "neutral")
            trending_hashtags = {
                "positive": ["#goodvibes", "#blessed", "#amazing", "#love"],
                "negative": ["#support", "#tough", "#real"],
                "neutral": ["#life", "#daily", "#moment", "#thoughts"]
            }
            
            for hashtag in trending_hashtags.get(sentiment, trending_hashtags["neutral"]):
                suggested_hashtags.append({
                    "hashtag": hashtag,
                    "category": "sentiment_based",
                    "relevance": round(sentiment_result.get("confidence", 0.5), 3),
                    "keyword": hashtag.replace("#", "")
                })
            
            # Sort by relevance and limit results
            suggested_hashtags.sort(key=lambda x: x["relevance"], reverse=True)
            top_hashtags = suggested_hashtags[:max_hashtags]
            
            # Add some general popular hashtags if we don't have enough
            if len(top_hashtags) < 5:
                popular_hashtags = ["#social", "#post", "#share", "#content", "#daily"]
                for hashtag in popular_hashtags:
                    if len(top_hashtags) >= max_hashtags:
                        break
                    if not any(h["hashtag"] == hashtag for h in top_hashtags):
                        top_hashtags.append({
                            "hashtag": hashtag,
                            "category": "popular",
                            "relevance": 0.5,
                            "keyword": hashtag.replace("#", "")
                        })
            
            self.logger.info(f"Generated {len(top_hashtags)} hashtag suggestions")
            
            return {
                "success": True,
                "hashtags": top_hashtags,
                "total_suggestions": len(top_hashtags),
                "categories_found": list(set(h["category"] for h in top_hashtags)),
                "text_analysis": {
                    "word_count": len(words),
                    "sentiment": sentiment,
                    "sentiment_confidence": sentiment_result.get("confidence", 0.5)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Hashtag suggestion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Hashtag suggestion error: {str(e)}")
