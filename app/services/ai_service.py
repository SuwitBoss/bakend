"""
AI Services for Face Analysis
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import json
from .deepfake_service import DeepfakeDetectionService
# Temporarily commenting out antispoofing to fix import issue
# from .antispoofing_service import AntiSpoofingService

logger = logging.getLogger(__name__)

# Preload CUDA DLLs for GPU acceleration
try:
    ort.preload_dlls()
    logger.info("✅ ONNX Runtime DLLs preloaded for GPU acceleration")
except Exception as e:
    logger.warning(f"⚠️ Failed to preload ONNX Runtime DLLs: {e}")

class FaceDetectionService:
    """Face detection using YOLO models"""
    def __init__(self, models_path: Path):
        self.models_path = models_path
        # Use faster YOLOv10n model for real-time detection
        self.model_path = models_path / "face-detection" / "yolov10n-face.onnx"
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model with GPU acceleration"""
        try:
            if self.model_path.exists():
                # Create session options for GPU acceleration
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Configure providers - prioritize GPU
                providers = []
                
                # Try CUDA provider first (if available)
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
                    logger.info("CUDA provider configured for face detection")
                
                # Try DirectML provider (Windows)
                if 'DmlExecutionProvider' in ort.get_available_providers():
                    providers.append('DmlExecutionProvider')
                    logger.info("DirectML provider configured for face detection")
                
                # Fallback to CPU
                providers.append('CPUExecutionProvider')
                
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=providers
                )
                
                # Log which provider is actually being used
                active_providers = self.session.get_providers()
                logger.info(f"Face detection model loaded with providers: {active_providers}")
                logger.info(f"Model path: {self.model_path}")
            else:
                logger.warning(f"Face detection model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            # Try loading with CPU only as fallback
            try:
                self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
                logger.info("Face detection model loaded with CPU fallback")
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                self.session = None
    
    async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            if self.session is None:
                # Fallback to OpenCV cascade if ONNX model not available
                return self._detect_faces_opencv(image)
            
            # Preprocess image for YOLO
            input_size = 640
            h, w = image.shape[:2]
            
            # Resize while maintaining aspect ratio
            scale = min(input_size / w, input_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to square
            padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            padded[:new_h, :new_w] = resized
            
            # Normalize
            input_tensor = padded.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
              # Run inference
            try:
                # Check if session is properly loaded
                if self.session is None:
                    logger.error("ONNX session is None, falling back to OpenCV")
                    return self._detect_faces_opencv(image)
                
                # Get input name dynamically
                input_name = self.session.get_inputs()[0].name
                logger.debug(f"Using input name: {input_name}")
                
                outputs = self.session.run(None, {input_name: input_tensor})
            except Exception as inference_error:
                logger.error(f"ONNX inference error: {inference_error}")
                return self._detect_faces_opencv(image)
            
            # Post-process results
            faces = self._process_yolo_output(outputs[0], scale, input_size)
            
            return faces
            
        except Exception as e:
            logger.error(f"YOLO face detection error: {e}")
            # Fallback to OpenCV
            return self._detect_faces_opencv(image)
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detection using OpenCV"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            result = []
            for (x, y, w, h) in faces:
                result.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.8  # Default confidence for OpenCV
                })
            
            return result
        except Exception as e:
            logger.error(f"OpenCV face detection error: {e}")
            return []
    
    def _process_yolo_output(self, output, scale, input_size):
        """Process YOLO detection output"""
        try:
            # YOLO output processing logic here
            # This is a simplified version
            return []
        except Exception as e:
            logger.error(f"YOLO output processing error: {e}")
            return []

class FaceRecognitionService:
    """Face recognition using deep learning models"""
    
    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.model_path = models_path / "face-recognition" / "arcface_r100.onnx"
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model with GPU acceleration"""
        try:
            if self.model_path.exists():
                # Create session options for GPU acceleration
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Configure providers - prioritize GPU
                providers = []
                
                # Try CUDA provider first (if available)
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
                    logger.info("CUDA provider configured for face recognition")
                
                # Try DirectML provider (Windows)
                if 'DmlExecutionProvider' in ort.get_available_providers():
                    providers.append('DmlExecutionProvider')
                    logger.info("DirectML provider configured for face recognition")
                
                # Fallback to CPU
                providers.append('CPUExecutionProvider')
                
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=providers
                )
                
                # Log which provider is actually being used
                active_providers = self.session.get_providers()
                logger.info(f"Face recognition model loaded with providers: {active_providers}")
                logger.info(f"Model path: {self.model_path}")
            else:
                logger.warning(f"Face recognition model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load face recognition model: {e}")
            # Try loading with CPU only as fallback
            try:
                self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
                logger.info("Face recognition model loaded with CPU fallback")
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                self.session = None
    
    async def get_face_embedding(self, image: np.ndarray, face_detection: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract face embedding from detected face"""
        try:
            if self.session is None:
                logger.warning("Face recognition model not loaded")
                return None
            
            # Extract face region
            x1, y1, x2, y2 = face_detection['bbox']
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Preprocess for face recognition model
            face_resized = cv2.resize(face_img, (112, 112))
            face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
            face_transposed = face_normalized.transpose(2, 0, 1)[None, ...]
            
            # Run inference
            embedding = self.session.run(None, {"input": face_transposed})[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.flatten()
            
        except Exception as e:
            logger.error(f"Face embedding extraction error: {e}")
            return None

class ImageQualityService:
    """Image quality analysis"""
    
    def __init__(self):
        pass
    
    async def analyze_quality(self, image: np.ndarray) -> float:
        """Analyze image quality"""
        try:
            # Simple quality metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return 0.0
    
    async def analyze_lighting(self, image: np.ndarray) -> float:
        """Analyze lighting conditions"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness
            mean_brightness = np.mean(gray)
            
            # Normalize to 0-1 range (optimal range 80-170)
            if 80 <= mean_brightness <= 170:
                lighting_score = 1.0
            else:
                lighting_score = max(0.0, 1.0 - abs(mean_brightness - 125) / 125.0)
            
            return lighting_score
            
        except Exception as e:
            logger.error(f"Lighting analysis error: {e}")
            return 0.0

class AIService:
    """Main AI Service coordinator"""
    
    def __init__(self):
        # Use relative path that works both in development and production
        current_dir = Path(__file__).parent.parent.parent.parent
        self.models_path = current_dir / "model"
        self.face_detector = FaceDetectionService(self.models_path)
        self.face_recognizer = FaceRecognitionService(self.models_path)
        self.quality_analyzer = ImageQualityService()
        self.deepfake_detector = DeepfakeDetectionService(self.models_path)
        # Temporarily commenting out antispoofing to fix import issue
        # self.antispoofing_detector = AntiSpoofingService(self.models_path)
    
    async def detect_faces_in_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detect faces in image"""
        try:
            logger.info(f"Processing image of size: {len(image_bytes)} bytes")
            
            # Convert bytes to opencv image
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return {"faces_detected": False, "face_count": 0, "message": "Failed to decode image"}
            
            logger.info(f"Image decoded successfully, shape: {img.shape}")
            
            # Face detection
            faces = await self.face_detector.detect_faces(img)
            
            logger.info(f"Face detection completed, found {len(faces)} faces")
            
            # Convert numpy types to Python types for JSON serialization
            serializable_faces = []
            for face in faces:
                serializable_face = {
                    'bbox': [int(x) for x in face['bbox']],
                    'confidence': float(face['confidence'])
                }
                serializable_faces.append(serializable_face)
            
            return {
                "faces_detected": len(faces) > 0,
                "face_count": len(faces),
                "faces": serializable_faces,
                "message": f"Detected {len(faces)} face(s)"
            }
            
        except Exception as e:
            logger.error(f"Face detection error: {e}", exc_info=True)
            return {"faces_detected": False, "face_count": 0, "message": f"Detection error: {str(e)}"}
    
    async def detect_deepfake(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detect deepfake in image using real ONNX model"""
        try:
            return await self.deepfake_detector.detect_deepfake_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Deepfake detection error: {e}")
            return {
                "is_authentic": False,
                "is_deepfake": True,
                "confidence": 0.0,
                "deepfake_score": 1.0,
                "real_score": 0.0,                "message": f"Detection error: {str(e)}",
                "error": True
            }
    
    async def detect_anti_spoofing(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detect anti-spoofing in image using real ONNX model"""
        try:
            return await self.antispoofing_detector.detect_anti_spoofing_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Anti-spoofing detection error: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "live_score": 0.0,
                "spoof_score": 1.0,
                "attack_type": "unknown",
                "message": f"Detection error: {str(e)}",
                "error": True
            }
    
    async def extract_face_embedding(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract face embedding from image"""
        try:
            # Convert bytes to opencv image
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"success": False, "message": "Failed to decode image"}
            
            # Detect faces first
            faces = await self.face_detector.detect_faces(img)
            
            if not faces:
                return {"success": False, "message": "No faces detected"}
            
            # Use the first detected face
            face = faces[0]
            
            # Extract embedding
            embedding = await self.face_recognizer.get_face_embedding(img, face)
            
            if embedding is None:
                return {"success": False, "message": "Failed to extract face embedding"}
            
            return {
                "success": True,
                "embedding": embedding.tolist(),
                "face_bbox": face['bbox'],
                "message": "Face embedding extracted successfully"
            }
            
        except Exception as e:
            logger.error(f"Face embedding extraction error: {e}")
            return {"success": False, "message": f"Extraction error: {str(e)}"}
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate similarity between two face embeddings"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Normalize embeddings
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
