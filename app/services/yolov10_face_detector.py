"""
YOLOv10 Face Detection Service - แทนที่ OpenCV Haar Cascade
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLOv10FaceDetector:
    """YOLOv10 face detection using ONNX model"""
    
    def __init__(self, model_path: str = "/app/model/face-detection/yolov10n-face.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_size = (640, 640)  # YOLOv10 input size        self.confidence_threshold = 0.3  # ลดลงเพื่อให้ตรวจจับได้ง่ายขึ้น
        self.nms_threshold = 0.4  # เพิ่มขึ้นเพื่อให้ overlap ได้มากขึ้น
        self.min_face_size = 30  # ลดขนาดขั้นต่ำ
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv10 ONNX model"""
        try:
            if Path(self.model_path).exists():
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                providers = ['CPUExecutionProvider']
                
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=sess_options,
                    providers=providers
                )
                
                logger.info(f"✅ YOLOv10 face detection model loaded successfully")
                
                # Log model info
                for input_meta in self.session.get_inputs():
                    logger.info(f"Input: {input_meta.name}, shape: {input_meta.shape}")
                for output_meta in self.session.get_outputs():
                    logger.info(f"Output: {output_meta.name}, shape: {output_meta.shape}")
                    
            else:
                logger.error(f"❌ YOLOv10 model not found: {self.model_path}")
                self.session = None
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv10 model: {e}")
            self.session = None
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for YOLOv10 inference"""
        orig_h, orig_w = image.shape[:2]
        
        # Calculate scale factor
        scale = min(self.input_size[0] / orig_w, self.input_size[1] / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image (gray background)
        padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.input_size[0] - new_w) // 2
        pad_y = (self.input_size[1] - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert to float and normalize [0, 1]
        processed = padded.astype(np.float32) / 255.0
        
        # Convert HWC to CHW and add batch dimension
        processed = np.transpose(processed, (2, 0, 1))
        processed = np.expand_dims(processed, axis=0)
        
        return processed, scale, (pad_x, pad_y)
    
    def _postprocess_detections(self, outputs: List[np.ndarray], scale: float, 
                              padding: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess YOLOv10 outputs to get face detections"""
        try:
            # YOLOv10 output format: [1, 300, 6] = [batch, detections, (x1, y1, x2, y2, conf, class)]
            predictions = outputs[0][0]  # Remove batch dimension
            
            faces = []
            pad_x, pad_y = padding
            orig_h, orig_w = orig_shape
            
            for detection in predictions:
                # Extract confidence (5th element, index 4)
                confidence = detection[4]
                
                if confidence > self.confidence_threshold:
                    # Extract bounding box (first 4 elements: x1, y1, x2, y2)
                    x1, y1, x2, y2 = detection[:4]
                    
                    # Adjust for padding and scale back to original image
                    x1 = (x1 - pad_x) / scale
                    y1 = (y1 - pad_y) / scale
                    x2 = (x2 - pad_x) / scale
                    y2 = (y2 - pad_y) / scale
                    
                    # Clip to image boundaries
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))                    # Calculate width and height
                    w = x2 - x1
                    h = y2 - y1
                      # เพิ่มการตรวจสอบขนาดขั้นต่ำและอัตราส่วนที่ผ่อนปรน
                    if w > self.min_face_size and h > self.min_face_size:
                        # ตรวจสอบ aspect ratio - ผ่อนปรนกว่าเดิม
                        aspect_ratio = w / h if h > 0 else 0
                        # ผ่อนปรนอัตราส่วนมากขึ้น
                        if 0.5 <= aspect_ratio <= 2.0:  # อัตราส่วนที่ผ่อนปรนขึ้น
                            # ตรวจสอบพื้นที่ใบหน้า - ขนาดขั้นต่ำ
                            face_area = w * h
                            if face_area > self.min_face_size * self.min_face_size * 0.5:  # ลดเงื่อนไขพื้นที่
                                faces.append({
                                    "x": int(x1),
                                    "y": int(y1),
                                    "width": int(w),
                                    "height": int(h),
                                    "confidence": float(confidence)
                                })            # Apply NMS to remove duplicate detections
            faces = self._apply_nms(faces)
            
            # Sort faces by confidence (highest first) และจำกัดจำนวน
            faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)            # จำกัดจำนวนใบหน้าสูงสุด (ปกติสำหรับการใช้งานทั่วไป)
            max_faces = 3  # อนุญาตให้ตรวจจับได้มากขึ้น
            if len(faces) > max_faces:
                logger.info(f"⚠️ ตรวจพบ {len(faces)} ใบหน้า แต่จำกัดแค่ {max_faces} ใบหน้าที่มี confidence สูงสุด")
                faces = faces[:max_faces]
              # ผ่อนปรนการกรองเพิ่มเติม - เลือกใบหน้าที่มี confidence ปานกลาง
            good_conf_faces = [f for f in faces if f["confidence"] > 0.3]  # ลดจาก 0.85 เป็น 0.3
            if len(good_conf_faces) > 0:
                faces = good_conf_faces
            
            # กรองใบหน้าที่มีขนาดปกติ
            normal_faces = []
            for face in faces:
                w, h = face["width"], face["height"]                # กรองใบหน้าที่เล็กเกินไปหรือใหญ่เกินไป (ผ่อนปรน)
                if 20 <= w <= 2000 and 20 <= h <= 2000:  # ผ่อนปรนขนาดมากขึ้น
                    # ตรวจสอบว่าไม่ใกล้ขอบรูปเกินไป (ผ่อนปรน)
                    if face["x"] >= 0 and face["y"] >= 0:  # ผ่อนปรนการตรวจสอบขอบ
                        normal_faces.append(face)
            
            faces = normal_faces
            return faces
            
        except Exception as e:
            logger.error(f"❌ Error in YOLOv10 postprocessing: {e}")
            return []
    
    def _apply_nms(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression"""
        if len(faces) <= 1:
            return faces
        
        boxes = []
        confidences = []
        
        for face in faces:
            boxes.append([face["x"], face["y"], face["width"], face["height"]])
            confidences.append(face["confidence"])
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        filtered_faces = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_faces.append(faces[i])
        
        return filtered_faces
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using YOLOv10"""
        if self.session is None:
            logger.warning("⚠️ YOLOv10 model not loaded, using OpenCV fallback")
            return self._opencv_fallback(image)
        
        try:
            orig_shape = image.shape[:2]  # (height, width)
            
            # Preprocess image
            processed_image, scale, padding = self._preprocess_image(image)
            
            # Get input name
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            outputs = self.session.run(None, {input_name: processed_image})
            
            # Postprocess results
            faces = self._postprocess_detections(outputs, scale, padding, orig_shape)
            
            logger.info(f"🎯 YOLOv10 detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"❌ YOLOv10 face detection error: {e}")
            return self._opencv_fallback(image)
    
    def _opencv_fallback(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback to OpenCV Haar Cascade"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            result = []
            for (x, y, w, h) in faces:
                result.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": 0.85  # Default confidence for OpenCV
                })
            
            logger.info(f"🔄 OpenCV fallback detected {len(result)} faces")
            return result
            
        except Exception as e:
            logger.error(f"❌ OpenCV fallback failed: {e}")
            return []
