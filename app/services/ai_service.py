"""
AI Services for Face Analysis
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os
import time
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
        
        # Check if model exists first
        if not self.model_path.exists():
            # Fallback to YOLOv5s if YOLOv10n is not available
            self.model_path = models_path / "face-detection" / "yolov5s-face.onnx"
            if not self.model_path.exists():
                logger.error(f"❌ No face detection models found in {models_path / 'face-detection'}")
        
        self.session = None
        self.input_name = None
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
                provider_options = {}
                
                # Try CUDA provider first (if available)
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    # Get GPU memory limit from settings
                    from app.core.config import settings
                    gpu_mem_limit = int(settings.VRAM_LIMIT_MB * 1024 * 1024)  # Convert to bytes
                    
                    cuda_provider_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': gpu_mem_limit,  # Use configured limit
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }
                    providers.append(('CUDAExecutionProvider', cuda_provider_options))
                    logger.info(f"🎮 CUDA provider configured for face detection with {gpu_mem_limit/(1024*1024):.0f}MB limit")
                
                # Try DirectML provider (Windows)
                elif 'DmlExecutionProvider' in ort.get_available_providers():
                    providers.append('DmlExecutionProvider')
                    logger.info("🪟 DirectML provider configured for face detection")
                
                # Fallback to CPU
                providers.append('CPUExecutionProvider')
                
                start_time = time.time()
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=providers
                )
                load_time = time.time() - start_time
                
                # Store input name for later use
                self.input_name = self.session.get_inputs()[0].name
                
                # Log which provider is actually being used
                active_providers = self.session.get_providers()
                logger.info(f"✅ Face detection model loaded in {load_time:.2f}s with providers: {active_providers}")
                logger.info(f"📂 Model path: {self.model_path}")
            else:
                logger.warning(f"⚠️ Face detection model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load face detection model: {e}")
            # Try loading with CPU only as fallback
            try:
                self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                logger.info("⚙️ Face detection model loaded with CPU fallback")
            except Exception as e2:
                logger.error(f"❌ CPU fallback also failed: {e2}")
                self.session = None
      async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            if self.session is None:
                # Fallback to OpenCV cascade if ONNX model not available
                logger.warning("⚠️ No ONNX session available, falling back to OpenCV")
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
                    logger.error("❌ ONNX session is None, falling back to OpenCV")
                    return self._detect_faces_opencv(image)
                
                # Use stored input name or get it dynamically
                input_name = self.input_name or self.session.get_inputs()[0].name
                
                # Perform inference
                start_time = time.time()
                outputs = self.session.run(None, {input_name: input_tensor})
                inference_time = time.time() - start_time
                
                logger.debug(f"🕒 YOLO inference completed in {inference_time*1000:.1f}ms")
            except Exception as inference_error:
                logger.error(f"❌ ONNX inference error: {inference_error}")
                return self._detect_faces_opencv(image)
            
            # Post-process results - properly implemented now
            faces = self._process_yolo_output(outputs[0], w, h, scale)
            logger.debug(f"🔍 Detected {len(faces)} faces with YOLO model")
            
            return faces
            
        except Exception as e:
            logger.error(f"❌ YOLO face detection error: {e}")
            # Fallback to OpenCV
            return self._detect_faces_opencv(image)
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback face detection using OpenCV"""
        try:
            # Try to use a pre-trained face cascade from OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            logger.info(f"📷 OpenCV fallback detected {len(faces)} faces")
            
            result = []
            for (x, y, w, h) in faces:
                result.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.8  # Default confidence for OpenCV
                })
            
            return result
        except Exception as e:
            logger.error(f"❌ OpenCV face detection error: {e}")
            return []
    
    def _process_yolo_output(self, output: np.ndarray, orig_width: int, orig_height: int, scale: float) -> List[Dict[str, Any]]:
        """Process YOLO detection output"""
        try:
            # Get output dimensions
            rows = output.shape[1]
            
            # Parse YOLO detections (newer YOLOv5/YOLOv8/YOLOv10 format)
            boxes = []
            confidences = []
            
            # Confidence threshold
            conf_threshold = 0.5
            
            # Process each detection
            for i in range(rows):
                row = output[0][i]
                confidence = row[4]
                
                # Only process detections with good confidence
                if confidence >= conf_threshold:
                    # Get class scores (after box coords and obj score)
                    classes_scores = row[5:]
                    
                    # We only care about the face class (usually class 0)
                    if classes_scores[0] > 0.5:
                        # Get bounding box coordinates
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        
                        # Convert to corner coordinates
                        x = cx - (w / 2)
                        y = cy - (h / 2)
                        
                        # Scale back to original image
                        input_size = 640  # YOLO input size
                        x = x * input_size / scale
                        y = y * input_size / scale
                        w = w * input_size / scale
                        h = h * input_size / scale
                        
                        # Add to results
                        boxes.append([int(x), int(y), int(x+w), int(y+h)])
                        confidences.append(float(confidence))
            
            # Convert to result format
            result = []
            for box, conf in zip(boxes, confidences):
                result.append({
                    'bbox': box,
                    'confidence': conf
                })
            
            return result
        except Exception as e:
            logger.error(f"❌ YOLO output processing error: {e}")
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
        # Use models_path from settings instead of hardcoded path
        from app.core.config import settings
        import os
        
        # Check if the model path exists and is properly mounted
        self.models_path = Path(settings.MODELS_PATH)
        if not self.models_path.exists():
            logger.error(f"❌ Model path not found: {self.models_path}")
            # Fallback to local development path if main path not found
            current_dir = Path(__file__).parent.parent.parent.parent
            self.models_path = current_dir / "model"
            logger.warning(f"⚠️ Using fallback model path: {self.models_path}")
        
        logger.info(f"📂 Using models path: {self.models_path}")
        
        # Initialize services with lazy loading
        self._face_detector = None
        self._face_recognizer = None
        self._deepfake_detector = None
        self._antispoofing_detector = None
        self._quality_analyzer = ImageQualityService()  # Lightweight, can initialize immediately
        
        # Track loaded models for memory management
        self._loaded_models = set()
        self._last_model_usage = {}
        
        # Set ONNX Runtime environment variables for better GPU performance
        os.environ["OMP_NUM_THREADS"] = str(settings.OMP_NUM_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(settings.OPENBLAS_NUM_THREADS) 
        os.environ["MKL_NUM_THREADS"] = str(settings.MKL_NUM_THREADS)
        os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"  # Reduce logging noise
        
        # Check available ONNX providers
        providers = ort.get_available_providers()
        logger.info(f"✅ Available ONNX Runtime providers: {providers}")
        
        # Pre-check for CUDA/GPU support
        self._has_cuda = 'CUDAExecutionProvider' in providers
        if self._has_cuda:
            logger.info("🚀 CUDA support is available for ONNX Runtime")
        else:
            logger.warning("⚠️ CUDA support not available, will use CPU for all models")
    
    @property
    def face_detector(self):
        """Lazy-loaded face detector"""
        if self._face_detector is None:
            self._unload_unused_models('face_detection')
            logger.info("🔄 Lazy-loading face detection model")
            self._face_detector = FaceDetectionService(self.models_path)
            self._loaded_models.add('face_detection')
        self._last_model_usage['face_detection'] = time.time()
        return self._face_detector
    
    @property
    def face_recognizer(self):
        """Lazy-loaded face recognizer"""
        if self._face_recognizer is None:
            self._unload_unused_models('face_recognition')
            logger.info("🔄 Lazy-loading face recognition model")
            self._face_recognizer = FaceRecognitionService(self.models_path)
            self._loaded_models.add('face_recognition')
        self._last_model_usage['face_recognition'] = time.time()
        return self._face_recognizer
    
    @property
    def deepfake_detector(self):
        """Lazy-loaded deepfake detector"""
        if self._deepfake_detector is None:
            self._unload_unused_models('deepfake_detection')
            logger.info("🔄 Lazy-loading deepfake detection model")
            self._deepfake_detector = DeepfakeDetectionService(self.models_path)
            self._loaded_models.add('deepfake_detection')
        self._last_model_usage['deepfake_detection'] = time.time()
        return self._deepfake_detector
    
    @property
    def antispoofing_detector(self):
        """Lazy-loaded antispoofing detector"""
        try:
            # Import here to avoid circular imports
            from .antispoofing_service import AntiSpoofingService
            
            if self._antispoofing_detector is None:
                self._unload_unused_models('antispoofing_detection')
                logger.info("🔄 Lazy-loading antispoofing detection model")
                self._antispoofing_detector = AntiSpoofingService(self.models_path)
                self._loaded_models.add('antispoofing_detection')
            self._last_model_usage['antispoofing_detection'] = time.time()
            return self._antispoofing_detector
        except ImportError as e:
            logger.error(f"❌ AntiSpoofing service import error: {e}")
            return None
    
    def _unload_unused_models(self, current_model: str, threshold_seconds: int = 300):
        """Unload models that haven't been used recently to save VRAM"""
        from app.core.config import settings
        import psutil
        
        # Only apply VRAM management in production with multiple models
        if settings.ENVIRONMENT != "production" or len(self._loaded_models) <= 1:
            return
        
        # Check if we need to perform memory management
        if len(self._loaded_models) >= 2:
            current_time = time.time()
            models_to_unload = []
            
            # Find models that haven't been used recently
            for model_name in self._loaded_models:
                if model_name == current_model:
                    continue
                
                last_used = self._last_model_usage.get(model_name, 0)
                if current_time - last_used > threshold_seconds:
                    models_to_unload.append(model_name)
            
            # Unload models to free memory
            for model_name in models_to_unload:
                logger.info(f"🧹 Unloading unused model: {model_name} to free memory")
                if model_name == 'face_detection' and self._face_detector is not None:
                    self._face_detector.session = None
                    self._face_detector = None
                elif model_name == 'face_recognition' and self._face_recognizer is not None:
                    self._face_recognizer.session = None
                    self._face_recognizer = None
                elif model_name == 'deepfake_detection' and self._deepfake_detector is not None:
                    self._deepfake_detector.session = None
                    self._deepfake_detector = None
                elif model_name == 'antispoofing_detection' and self._antispoofing_detector is not None:
                    self._antispoofing_detector.session = None
                    self._antispoofing_detector = None
                
                self._loaded_models.remove(model_name)
            
            # Log memory usage after unloading
            if models_to_unload:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"💾 Memory usage after unloading: {memory_info.rss / (1024 * 1024):.2f} MB")

    # Rest of the methods stay the same with small adjustments
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
            
            # Face detection - use property to ensure lazy loading
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
            # Use property to ensure lazy loading
            return await self.deepfake_detector.detect_deepfake_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Deepfake detection error: {e}")
            return {
                "is_authentic": False,
                "is_deepfake": True,
                "confidence": 0.0,
                "deepfake_score": 1.0,
                "real_score": 0.0,                
                "message": f"Detection error: {str(e)}",
                "error": True
            }
    
    async def detect_anti_spoofing(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detect anti-spoofing in image using real ONNX model"""
        try:
            # Use property to ensure lazy loading
            detector = self.antispoofing_detector
            if detector is None:
                raise ImportError("Anti-spoofing detector not available")
            return await detector.detect_anti_spoofing_from_bytes(image_bytes)
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
            
            # Detect faces first - use property for lazy loading
            faces = await self.face_detector.detect_faces(img)
            
            if not faces:
                return {"success": False, "message": "No faces detected"}
            
            # Use the first detected face
            face = faces[0]
            
            # Extract embedding - use property for lazy loading
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
