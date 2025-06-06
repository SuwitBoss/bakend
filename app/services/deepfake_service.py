"""
Deepfake Detection Service using ONNX model
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DeepfakeDetectionService:
    """Deepfake detection using Xception ONNX model"""
    
    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.model_path = models_path / "deepfake-detection" / "model.onnx"
        self.session = None
        self.input_size = (299, 299)  # Xception model input size
        self._load_model()
        
    def _load_model(self):
        """Load ONNX model with GPU acceleration when available"""
        try:
            if self.model_path.exists():
                # Create session options for optimization
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Configure threading for multi-core performance
                sess_options.intra_op_num_threads = 0  # 0 = use all available cores
                sess_options.inter_op_num_threads = 0  # 0 = use all available cores
                sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                
                # Configure providers - prioritize GPU
                providers = []
                
                # Try CUDA provider first (if available)
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 1GB limit for deepfake detection
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
                    logger.info("CUDA provider configured for deepfake detection")
                
                # Fallback to CPU
                providers.append('CPUExecutionProvider')
                logger.info("Deepfake detection service configured with providers: " + str(providers))
                
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=providers
                )
                
                # Log which provider is actually being used
                active_providers = self.session.get_providers()
                logger.info(f"Deepfake detection model loaded with providers: {active_providers}")
                
                # Log model input/output info
                for input_meta in self.session.get_inputs():
                    logger.info(f"Input: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
                for output_meta in self.session.get_outputs():
                    logger.info(f"Output: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")
            else:
                logger.warning(f"Deepfake detection model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load deepfake detection model: {e}")
            # Try loading with CPU only as fallback
            try:
                self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
                logger.info("Deepfake detection model loaded with CPU fallback")
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                self.session = None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Xception model"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Resize to model input size (299x299 for Xception)
            pil_image = pil_image.resize(self.input_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            
            # Normalize to [-1, 1] range (as per Xception model requirements)
            image_array = (image_array - 0.5) / 0.5
            
            # Change shape from (299, 299, 3) to (1, 3, 299, 299)
            image_array = np.transpose(image_array, (2, 0, 1))
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    def _get_input_name(self) -> str:
        """Get the input tensor name from the model"""
        if self.session:
            return self.session.get_inputs()[0].name
        return "pixel_values"  # Default fallback
    
    async def detect_deepfake(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect deepfake in image using ONNX model"""
        try:
            if self.session is None:
                logger.warning("Deepfake detection model not loaded, using fallback")
                return self._fallback_detection()
            
            # Preprocess image
            input_data = self._preprocess_image(image)
            
            # Get input name
            input_name = self._get_input_name()
            
            # Run inference
            outputs = self.session.run(None, {input_name: input_data})
            logits = outputs[0]
            
            # Calculate probabilities using softmax
            probabilities = self._softmax(logits[0])
            
            # Class mapping: 0 = Deepfake, 1 = Real
            deepfake_prob = float(probabilities[0])
            real_prob = float(probabilities[1])
            
            # Determine result
            is_deepfake = deepfake_prob > real_prob
            confidence = max(deepfake_prob, real_prob)
            
            # Create result
            result = {
                "is_deepfake": is_deepfake,
                "is_authentic": not is_deepfake,
                "confidence": confidence,
                "deepfake_score": deepfake_prob,
                "real_score": real_prob,
                "message": "Deepfake detected" if is_deepfake else "Image appears authentic",
                "model_info": {
                    "model_type": "Xception",
                    "input_size": self.input_size,
                    "threshold": 0.5
                }
            }
            
            logger.info(f"Deepfake detection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Deepfake detection error: {e}")
            return {
                "is_deepfake": False,
                "is_authentic": False,
                "confidence": 0.0,
                "deepfake_score": 0.0,
                "real_score": 0.0,
                "message": f"Detection error: {str(e)}",
                "error": True
            }
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def _fallback_detection(self) -> Dict[str, Any]:
        """Fallback detection when model is not available"""
        return {
            "is_deepfake": False,
            "is_authentic": True,
            "confidence": 0.5,
            "deepfake_score": 0.2,
            "real_score": 0.8,
            "message": "Model not available - using fallback detection",
            "fallback": True
        }
    
    async def detect_deepfake_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detect deepfake from image bytes"""
        try:
            # Convert bytes to opencv image
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "is_deepfake": False,
                    "is_authentic": False,
                    "confidence": 0.0,
                    "deepfake_score": 0.0,
                    "real_score": 0.0,
                    "message": "Failed to decode image",
                    "error": True
                }
            
            return await self.detect_deepfake(img)
            
        except Exception as e:
            logger.error(f"Deepfake detection from bytes error: {e}")
            return {
                "is_deepfake": False,
                "is_authentic": False,
                "confidence": 0.0,
                "deepfake_score": 0.0,
                "real_score": 0.0,
                "message": f"Detection error: {str(e)}",
                "error": True
            }
