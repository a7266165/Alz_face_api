"""
MiVOLO v2 年齡預測器

移植自 Alz_face_analyze/src/age/predictor/mivolo.py：predict_single 直接把
（對齊後的全幀）影像交給 HF AutoImageProcessor，不另做 Haar 裁臉——與訓練端
predicted_ages.json 的產生方式一致。多張聚合取「平均」（對齊 load_predicted_ages
的 mean 慣例，非中位數）。
"""

import os

# 在 import transformers 之前停用 TensorFlow 後端並壓低 log。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MiVOLOPredictor:
    """MiVOLO v2 年齡預測器。"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def initialize(self) -> None:
        """載入模型（HuggingFace iitolstykh/mivolo_v2）。"""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch

            use_cuda = torch.cuda.is_available()
            self._device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32

            self._model = AutoModelForImageClassification.from_pretrained(
                "iitolstykh/mivolo_v2", trust_remote_code=True, dtype=dtype
            )
            self._processor = AutoImageProcessor.from_pretrained(
                "iitolstykh/mivolo_v2", trust_remote_code=True
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"✓ MiVOLO 初始化完成 ({self._device.upper()})")
        except Exception as e:
            raise RuntimeError(f"MiVOLO 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """單張（對齊全幀）影像 → 年齡；失敗回 None。"""
        import torch

        try:
            inputs = self._processor(images=[image])["pixel_values"]
            inputs = inputs.to(dtype=self._model.dtype, device=self._model.device)
            with torch.no_grad():
                outputs = self._model(faces_input=inputs, body_input=inputs)
            if hasattr(outputs, "age_output"):
                return float(outputs.age_output[0].item())
            logger.debug("MiVOLO 輸出缺 age_output（schema 不符？）")
        except Exception as e:
            logger.debug(f"年齡預測失敗: {e}")
        return None

    def predict(self, images: List[np.ndarray]) -> Optional[float]:
        """多張影像 → 年齡「平均」（與訓練端 load_predicted_ages 一致）；全失敗回 None。"""
        ages = [a for img in images if (a := self.predict_single(img)) is not None]
        if not ages:
            return None
        return float(np.mean(ages))
