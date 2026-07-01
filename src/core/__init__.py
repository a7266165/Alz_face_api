"""
核心模組：與 Alz_face_analyze 訓練端對齊的預處理 / 特徵 / 年齡。

公開 API：
    from src.core import (
        PipelineConfig, MirrorConfig,
        FacePreprocessor, ProcessedFace,
        ArcFaceExtractor, MiVOLOPredictor,
    )
"""

from .config import PipelineConfig, MirrorConfig
from .preprocess import FacePreprocessor, ProcessedFace, FaceInfo
from .feature_extract import ArcFaceExtractor
from .age_predictor import MiVOLOPredictor

__version__ = "2.0.0"

__all__ = [
    "PipelineConfig",
    "MirrorConfig",
    "FacePreprocessor",
    "ProcessedFace",
    "FaceInfo",
    "ArcFaceExtractor",
    "MiVOLOPredictor",
]
