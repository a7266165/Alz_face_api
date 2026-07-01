"""
ArcFace 特徵提取器（InsightFace buffalo_l，輸出 512 維）。

移植自 Alz_face_analyze/src/embedding/extractor/arcface.py，逐字一致：
回傳 InsightFace 偵測到的 raw embedding（未額外 L2 正規化；正規化交給下游
LR pipeline 內的 StandardScaler）。偵測失敗時退回整張圖縮放後直接過辨識模型。
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ArcFaceExtractor:
    """ArcFace 512 維人臉特徵提取器（InsightFace buffalo_l）。"""

    def __init__(self):
        self._app = None

    def initialize(self) -> None:
        """載入 InsightFace FaceAnalysis（buffalo_l 偵測器 + ArcFace 辨識模型）。"""
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._app = app
        logger.info("✓ ArcFace (buffalo_l) 初始化完成")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """提取單張 ArcFace 512 維特徵（BGR 輸入）。"""
        if self._app is None:
            raise RuntimeError("ArcFaceExtractor 尚未 initialize()")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self._app.get(image_rgb)

        if not faces:
            logger.debug("ArcFace 未偵測到人臉，改用整張圖")
            img_resized = cv2.resize(image_rgb, (112, 112))
            img_input = (
                np.transpose(img_resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32) - 127.5
            ) / 127.5
            embedding = self._app.models["recognition"].forward(img_input)
            return embedding.flatten().astype(np.float32)

        return faces[0].embedding.astype(np.float32)

    def extract_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """逐張提取；單張失敗回 None（保留位置以利配對過濾）。"""
        out: List[Optional[np.ndarray]] = []
        for i, image in enumerate(images):
            try:
                out.append(self.extract(image))
            except Exception as e:
                logger.warning(f"ArcFace 第 {i} 張提取失敗: {e}")
                out.append(None)
        return out
