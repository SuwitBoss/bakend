"""
Anti-spoofing Service for FaceSocial Backend
This service provides face anti-spoofing detection based on the Face-AntiSpoofing model.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List
import os
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)

class AntispoofingService:
    """Anti-spoofing detection service using Face-AntiSpoofing models"""
    
    def __init__(self):
        """Initialize anti-spoofing service"""
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.models_path = None
        self.binary_model = None
        self.print_replay_model = None
        self.model_img_size = 128
        
    async def initialize(self, models_path: Path = None):
        """Initialize anti-spoofing models"""
        if self.initialized:
            return
            
        try:
            # Set models path
            if models_path is None:
                from app.core.config import settings
                models_path = Path(settings.MODELS_PATH) / "anti-spoofing"
            else:
                models_path = Path(models_path)
                
            self.models_path = models_path
            
            # Load binary classification model
            binary_model_path = models_path / "AntiSpoofing_bin_1.5_128.onnx"
            self.binary_model = await self._load_model(binary_model_path)
            
            # Load print-replay model
            print_replay_model_path = models_path / "AntiSpoofing_print-replay_1.5_128.onnx"
            self.print_replay_model = await self._load_model(print_replay_model_path)
            
            self.initialized = True
            self.logger.info("✅ Anti-spoofing service initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize anti-spoofing service: {e}")
            raise
    
    async def _load_model(self, model_path: Path):
        """Load ONNX model with appropriate providers"""
        if not model_path.exists():
            self.logger.error(f"❌ Model not found: {model_path}")
            return None
            
        try:
            # Configure providers - prioritize GPU
            providers = []
            
            # Try CUDA provider first (if available)
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                self.logger.info(f"🎮 Using CUDA for anti-spoofing model: {model_path.name}")
            
            # Fallback to CPU
            providers.append('CPUExecutionProvider')
            
            # Create session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Load model
            model = {
                'session': ort.InferenceSession(str(model_path), providers=providers, sess_options=sess_options),
                'path': model_path
            }
            
            # Store input name
            model['input_name'] = model['session'].get_inputs()[0].name
            
            self.logger.info(f"✅ Loaded anti-spoofing model: {model_path.name}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load anti-spoofing model: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for anti-spoofing model"""
        # Resize while maintaining aspect ratio
        new_size = self.model_img_size
        old_size = image.shape[:2]  # (height, width)
        
        ratio = float(new_size) / max(old_size)
        scaled_shape = tuple([int(x * ratio) for x in old_size])
        
        # Resize image
        resized = cv2.resize(image, (scaled_shape[1], scaled_shape[0]))
        
        # Add padding to make square
        delta_w = new_size - scaled_shape[1]
        delta_h = new_size - scaled_shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Normalize and transpose for ONNX input
        processed = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Apply softmax to prediction"""
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))
            
        return softmax(prediction)
    
    def _increased_crop(self, image: np.ndarray, bbox: Dict[str, int], bbox_inc: float = 1.5) -> np.ndarray:
        """
        Create increased bounding box crop around face
        
        Args:
            image: Input image
            bbox: Dictionary with x, y, width, height
            bbox_inc: Bounding box increase factor
            
        Returns:
            Cropped image with padding
        """
        real_h, real_w = image.shape[:2]
        
        x, y = bbox['x'], bbox['y']
        w, h = bbox['width'], bbox['height']
        l = max(w, h)
        
        xc, yc = x + w / 2, y + h / 2
        x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
        
        x1 = 0 if x < 0 else x 
        y1 = 0 if y < 0 else y
        x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
        y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
        
        cropped = image[y1:y2, x1:x2, :]
        
        # Add padding if necessary
        padded = cv2.copyMakeBorder(
            cropped,
            y1 - y, int(l * bbox_inc - y2 + y),
            x1 - x, int(l * bbox_inc) - x2 + x,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        return padded
    
    async def detect_spoofing(self, image: np.ndarray, face_bbox: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Detect face spoofing attempts
        
        Args:
            image: Input image as numpy array
            face_bbox: Face bounding box dictionary with x, y, width, height
                       If None, the entire image is used
            
        Returns:
            Spoofing detection results with confidence score
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Ensure image is RGB
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                # Already RGB, do nothing
                pass
            else:
                self.logger.warning(f"⚠️ Unexpected image format: {image.shape}, {image.dtype}")
            
            # If face bounding box is provided, crop the image
            if face_bbox is not None:
                image = self._increased_crop(image, face_bbox, bbox_inc=1.5)
            
            # Make sure binary model is loaded
            if self.binary_model is None:
                return {
                    "is_real": False,
                    "confidence": 0.0,
                    "method": "error",
                    "error": "Binary model not loaded"
                }
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run binary model
            binary_result = self.binary_model['session'].run(
                [], {self.binary_model['input_name']: processed_image}
            )
            binary_pred = self._postprocess_prediction(binary_result[0])
            binary_score = float(binary_pred[0][0])  # Real face probability
            binary_is_real = binary_score > 0.5
            
            result = {
                "is_real": binary_is_real,
                "confidence": binary_score,
                "method": "binary"
            }
            
            # If print-replay model is available, add its prediction
            if self.print_replay_model is not None:
                try:
                    pr_result = self.print_replay_model['session'].run(
                        [], {self.print_replay_model['input_name']: processed_image}
                    )
                    pr_pred = self._postprocess_prediction(pr_result[0])
                    
                    # Add print-replay predictions
                    # Class 0: Real, Class 1: Print Attack, Class 2: Replay Attack
                    result["print_attack_score"] = float(pr_pred[0][1])
                    result["replay_attack_score"] = float(pr_pred[0][2])
                    result["attack_type"] = "none" if binary_is_real else (
                        "print" if pr_pred[0][1] > pr_pred[0][2] else "replay"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Error running print-replay model: {e}")
            
            self.logger.info(f"🔍 Anti-spoofing result: {'REAL' if binary_is_real else 'FAKE'} with {binary_score:.2f} confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Anti-spoofing detection error: {e}")
            return {
                "is_real": False,
                "confidence": 0.0,
                "method": "error",
                "error": str(e)
            }