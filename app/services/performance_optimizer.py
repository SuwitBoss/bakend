from functools import lru_cache
import gc
import torch
from typing import List, Dict, Any
import asyncio

class PerformanceOptimizer:
    """Performance optimization utilities for AI workloads"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def preprocess_image(image_hash: str, image_data: bytes):
        """Cache preprocessed images to avoid redundant processing"""
        # Image preprocessing logic
        return processed_image
    
    @staticmethod
    def cleanup_memory():
        """Periodic memory cleanup for long-running processes"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    async def batch_process_images(images: List[bytes], process_single_image) -> List[Dict]:
        """Process multiple images in batches for efficiency"""
        batch_size = 4
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                process_single_image(img) for img in batch
            ])
            results.extend(batch_results)
            
            # Cleanup memory after each batch
            PerformanceOptimizer.cleanup_memory()
        
        return results
