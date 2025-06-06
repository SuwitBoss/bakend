import onnxruntime as ort
from typing import Dict, Any
import asyncio
from functools import lru_cache

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
            cls._instance.initialized = False
        return cls._instance
    
    async def initialize(self):
        """Initialize all AI models with optimized ONNX Runtime configuration"""
        if self.initialized:
            return
            
        # Optimized ONNX Runtime providers
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            ('CPUExecutionProvider', {
                'intra_op_num_threads': 4,
                'inter_op_num_threads': 2,
            })
        ]
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Load all models
        model_configs = {
            'face_detection': '/app/models/yolo_face.onnx',
            'anti_spoofing': '/app/models/antispoofing.onnx', 
            'gender_age': '/app/models/genderage.onnx',
            'deepfake': '/app/models/deepfake.onnx'
        }
        
        for model_name, model_path in model_configs.items():
            try:
                self.models[model_name] = ort.InferenceSession(
                    model_path, 
                    providers=providers,
                    sess_options=session_options
                )
                print(f"Loaded {model_name} model successfully")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        self.initialized = True
    
    async def analyze_face_pipeline(self, image_data: bytes) -> Dict[str, Any]:
        """Complete face analysis pipeline with parallel processing"""
        # 1. Face detection (runs once)
        faces = await self.detect_faces(image_data)
        
        if not faces:
            return {"error": "No faces detected"}
        
        # 2. Parallel analysis of detected faces
        face_crop = faces[0]  # Process first face
        
        tasks = [
            self.analyze_anti_spoofing(face_crop),
            self.analyze_gender_age(face_crop),
            self.analyze_deepfake(face_crop)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "faces_detected": len(faces),
            "anti_spoofing": results[0] if not isinstance(results[0], Exception) else None,
            "demographics": results[1] if not isinstance(results[1], Exception) else None,
            "deepfake": results[2] if not isinstance(results[2], Exception) else None
        }
