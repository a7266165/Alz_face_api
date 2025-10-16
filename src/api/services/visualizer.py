"""
src/api/services/visualizer.py
人臉標記圖片生成服務
"""

import logging
import base64
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


class FaceVisualizer:
    """人臉視覺化標記器"""
    
    # 人臉中軸線
    FACEMESH_MID_LINE = [
        (10, 151), (151, 9), (9, 8), (8, 168), (168, 6),
        (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
        (1, 19), (19, 94), (94, 2),
    ]
    
    def __init__(self):
        """初始化視覺化標記器"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
        )
        logger.info("✓ MediaPipe FaceMesh 初始化完成")
    
    def generate_marked_image(self, image: np.ndarray) -> Optional[str]:
        """
        生成標記圖片
        
        Args:
            image: 原始圖片（BGR 格式）
            
        Returns:
            Base64 編碼的標記圖片，失敗則回傳 None
        """
        try:
            # 轉換為 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 偵測人臉
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                logger.warning("未偵測到人臉")
                return None
            
            # 標記圖片
            marked_image = self._draw_landmarks(image.copy(), results)
            
            # 轉換為 base64
            base64_str = self._encode_base64(marked_image)
            
            logger.info("✓ 標記圖片生成成功")
            return base64_str
            
        except Exception as e:
            logger.error(f"生成標記圖片失敗: {e}")
            return None
    
    def _draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        在圖片上標記 landmarks
        
        Args:
            image: 原始圖片
            results: MediaPipe 偵測結果
            
        Returns:
            標記後的圖片
        """
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 繪製所有 landmarks（綠色小圓點）
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # 繪製中軸線（紅色）
        for start_idx, end_idx in self.FACEMESH_MID_LINE:
            start_pt = landmarks[start_idx]
            end_pt = landmarks[end_idx]
            
            start = (int(start_pt.x * w), int(start_pt.y * h))
            end = (int(end_pt.x * w), int(end_pt.y * h))
            
            cv2.line(image, start, end, (0, 0, 255), 2)
        
        return image
    
    def _encode_base64(self, image: np.ndarray) -> str:
        """
        將圖片編碼為 base64
        
        Args:
            image: 圖片（BGR 格式）
            
        Returns:
            Base64 編碼字串（含 data URI prefix）
        """
        # 編碼為 JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        # 轉換為 base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 加上 data URI prefix
        return f"data:image/jpeg;base64,{img_base64}"
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()