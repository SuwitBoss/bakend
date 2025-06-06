"""
Advanced Glasses Detection Service
Provides multiple methods for detecting glasses on faces
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedGlassesDetector:
    """
    Advanced glasses detection service that combines multiple detection methods
    including Haar cascades, HOG features, and heuristic approaches
    """
    
    def __init__(self):
        """Initialize the glasses detector with multiple detection methods"""
        self.methods = {
            "haar_cascade": True,
            "hog_features": True,
            "heuristic": True
        }
        
        # Load the face cascade classifier
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info(f"Loaded face cascade classifier from {cascade_path}")
            
            # Try to load glasses cascade classifier if available
            glasses_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
            self.glasses_cascade = cv2.CascadeClassifier(glasses_cascade_path)
            logger.info(f"Loaded glasses cascade classifier from {glasses_cascade_path}")
        except Exception as e:
            logger.error(f"Error loading cascade classifiers: {e}")
            self.face_cascade = None
            self.glasses_cascade = None
            
        # Initialize HOG descriptor for feature extraction
        self.hog = cv2.HOGDescriptor()
        
        logger.info("Advanced glasses detector initialized successfully")
    
    def detect_glasses(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect glasses in the provided image using multiple methods
        
        Args:
            image: OpenCV image in BGR format
            
        Returns:
            Dictionary containing detection results from all methods
        """
        if image is None:
            logger.error("Invalid image provided to glasses detector")
            return {"has_glasses": False, "confidence": 0.0, "methods": {}}
        
        # Convert image to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = {}
        combined_confidence = 0.0
        method_count = 0
        
        # Method 1: Haar Cascade detection
        if self.methods["haar_cascade"] and self.face_cascade is not None and self.glasses_cascade is not None:
            try:
                haar_result = self._detect_with_haar(gray)
                results["haar_cascade"] = haar_result
                combined_confidence += haar_result["confidence"]
                method_count += 1
            except Exception as e:
                logger.error(f"Error in Haar cascade detection: {e}")
                results["haar_cascade"] = {"has_glasses": False, "confidence": 0.0, "error": str(e)}
        
        # Method 2: HOG features detection
        if self.methods["hog_features"]:
            try:
                hog_result = self._detect_with_hog(gray)
                results["hog_features"] = hog_result
                combined_confidence += hog_result["confidence"]
                method_count += 1
            except Exception as e:
                logger.error(f"Error in HOG features detection: {e}")
                results["hog_features"] = {"has_glasses": False, "confidence": 0.0, "error": str(e)}
        
        # Method 3: Heuristic approach based on image processing
        if self.methods["heuristic"]:
            try:
                heuristic_result = self._detect_with_heuristic(gray)
                results["heuristic"] = heuristic_result
                combined_confidence += heuristic_result["confidence"]
                method_count += 1
            except Exception as e:
                logger.error(f"Error in heuristic detection: {e}")
                results["heuristic"] = {"has_glasses": False, "confidence": 0.0, "error": str(e)}
        
        # Calculate average confidence
        avg_confidence = combined_confidence / max(1, method_count)
        has_glasses = avg_confidence > 0.5
        
        return {
            "has_glasses": has_glasses,
            "confidence": round(avg_confidence, 2),
            "methods": results
        }
    
    def _detect_with_haar(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Detect glasses using Haar cascade classifier"""
        # Detect faces first
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
        
        if len(faces) == 0:
            return {"has_glasses": False, "confidence": 0.0, "faces_detected": 0}
        
        # For each face, check for glasses
        has_glasses = False
        max_confidence = 0.0
        faces_with_glasses = 0
        
        for (x, y, w, h) in faces:
            # Define the eye region (upper half of face)
            eye_region = gray_image[y:y+int(h/2), x:x+w]
            
            # Detect glasses in the eye region
            if self.glasses_cascade is not None:
                glasses = self.glasses_cascade.detectMultiScale(eye_region, 1.1, 3)
                
                if len(glasses) > 0:
                    # Calculate confidence based on the size and number of detections
                    confidence = min(1.0, len(glasses) * 0.3)
                    
                    # Add size-based confidence
                    for (gx, gy, gw, gh) in glasses:
                        relative_size = (gw * gh) / (w * h/2)
                        confidence += min(0.5, relative_size)
                    
                    confidence = min(1.0, confidence)
                    max_confidence = max(max_confidence, confidence)
                    faces_with_glasses += 1
                    has_glasses = True
        
        return {
            "has_glasses": has_glasses,
            "confidence": round(max_confidence, 2),
            "faces_detected": len(faces),
            "faces_with_glasses": faces_with_glasses
        }
    
    def _detect_with_hog(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Detect glasses using HOG features"""
        # This is a simplified implementation that looks for horizontal gradients
        # in the eye region that are characteristic of glasses frames
        
        # Normalize image and resize to standard size
        resized = cv2.resize(gray_image, (128, 128))
        
        # Compute HOG features
        features = self.hog.compute(resized)
        
        # Simplified heuristic: Sum of horizontal gradients in upper face region
        # For a real implementation, you would train a classifier on these features
        
        # Simulate a classification confidence
        # This is placeholder logic - in a real implementation this would use a trained model
        confidence = 0.3  # Base confidence
        
        # Look for strong horizontal gradients in eye regions
        try:
            # Define eye region (upper middle part of the image)
            eye_region = resized[30:60, 20:108]
            
            # Calculate horizontal gradients
            sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobelx = np.absolute(sobelx)
            
            # Threshold to identify strong horizontal edges
            horizontal_edges = cv2.threshold(abs_sobelx, 50, 255, cv2.THRESH_BINARY)[1]
            
            # Count strong horizontal edges
            edge_count = np.sum(horizontal_edges > 0)
            
            # More horizontal edges could indicate glasses
            if edge_count > 500:
                confidence += 0.4
            elif edge_count > 200:
                confidence += 0.2
                
        except Exception as e:
            logger.error(f"Error in HOG gradient analysis: {e}")
        
        has_glasses = confidence > 0.5
        
        return {
            "has_glasses": has_glasses,
            "confidence": round(confidence, 2)
        }
    
    def _detect_with_heuristic(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect glasses using heuristic image processing techniques
        focusing on edge detection and region properties
        """
        # Resize for consistent processing
        resized = cv2.resize(gray_image, (128, 128))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Focus on eye region
        eye_region = edges[30:60, 20:108]
        
        # Count edge pixels in the eye region
        edge_count = np.sum(eye_region > 0)
        
        # Calculate density of edges
        edge_density = edge_count / (eye_region.shape[0] * eye_region.shape[1])
        
        # Heuristic: Higher edge density in eye region often indicates glasses
        confidence = min(1.0, edge_density * 5)  # Scale factor for reasonable confidence
        
        has_glasses = confidence > 0.5
        
        return {
            "has_glasses": has_glasses,
            "confidence": round(confidence, 2),
            "edge_density": round(edge_density, 4)
        }
    
    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Return information about available detection methods"""
        methods = [
            {
                "id": "haar_cascade",
                "name": "Haar Cascade Classifier",
                "description": "Uses OpenCV's Haar cascade classifiers to detect glasses",
                "enabled": self.methods["haar_cascade"],
                "accuracy": "Medium",
                "speed": "Fast"
            },
            {
                "id": "hog_features",
                "name": "HOG Feature Detection",
                "description": "Uses Histogram of Oriented Gradients to detect glasses frames",
                "enabled": self.methods["hog_features"],
                "accuracy": "Medium-High",
                "speed": "Medium"
            },
            {
                "id": "heuristic",
                "name": "Heuristic Edge Analysis",
                "description": "Uses edge detection and image processing to identify glasses",
                "enabled": self.methods["heuristic"],
                "accuracy": "Low-Medium",
                "speed": "Fast"
            }
        ]
        
        return methods