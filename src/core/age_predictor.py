"""
src/core/age_predictor.py
MiVOLO 年齡預測器
"""

import os
# 在 import transformers 之前禁用 tensorflow
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from typing import List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MiVOLOPredictor:
    """MiVOLO v2 年齡預測器"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.face_detector = None
        self.device = None
    
    def initialize(self):
        """載入模型"""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForImageClassification.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True,
                dtype=dtype
            )
            self.processor = AutoImageProcessor.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True
            )
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info(f"✓ MiVOLO 初始化完成 ({self.device.upper()})")
            
        except Exception as e:
            raise RuntimeError(f"MiVOLO 初始化失敗: {e}")
    
    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """預測單張影像的年齡"""
        import torch
        
        try:
            # 偵測人臉
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                face_crop = image
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                margin = int(max(w, h) * 0.3)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
                face_crop = image[y1:y2, x1:x2]
            
            # 預處理
            inputs = self.processor(images=[face_crop])["pixel_values"]
            inputs = inputs.to(dtype=self.model.dtype, device=self.model.device)
            
            # 推論
            with torch.no_grad():
                outputs = self.model(faces_input=inputs, body_input=inputs)
            
            if hasattr(outputs, 'age_output'):
                return outputs.age_output[0].item()
                
        except Exception as e:
            logger.debug(f"預測失敗: {e}")
        
        return None
    
    def predict(self, images: List[np.ndarray]) -> Optional[float]:
        """
        預測多張影像的年齡（取中位數）
        
        Args:
            images: BGR 影像列表
            
        Returns:
            預測年齡（中位數），全部失敗則回傳 None
        """
        ages = []
        for img in images:
            age = self.predict_single(img)
            if age is not None:
                ages.append(age)
        
        if ages:
            return float(np.median(ages))
        return None