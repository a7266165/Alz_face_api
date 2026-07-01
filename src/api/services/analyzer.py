"""
主分析服務：串接新版 6 步 pipeline

(1) 輸入 BGR 影像列表 + real_age
(2) 預處理：選 10 張最正面 → 對齊 → 年齡臉特寫(age_crop) + flip 鏡射
(3) 年齡：MiVOLO 對 10 張 age_crop（landmark 外接框+margin 裁切）→ predicted_age（平均）；age_error = real − predicted
(4) ArcFace：20 張 → E_o(10×512) / E_m(10×512)
    (4.1) LR#1（10 折 base 集成）對 E_o → embedding_LR_score
    (4.2) D = E_o − E_m（differences）→ LR#2（10 折 base 集成）→ asymmetry_LR_score
(5) TabPFN(core3 = [embedding_LR_score, asymmetry_LR_score, age_error]) → ad_prob
(6) 回傳 { predicted_age, embedding_LR_score, ad_prob }
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.core import (
    ArcFaceExtractor,
    FacePreprocessor,
    MiVOLOPredictor,
    PipelineConfig,
)
from src.api.schemas import AnalysisResponse
from src.api.services.classifiers import FoldEnsembleScorer, TabPFNScorer

logger = logging.getLogger(__name__)


class AnalysisService:
    """主分析服務（啟動時載入所有模型）。"""

    def __init__(
        self,
        lr_embedding_folds_dir: Path,
        lr_asymmetry_folds_dir: Path,
        tabpfn_path: Path,
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig()

        # 特徵 / 年齡（重模型，載入一次）
        self.arcface = ArcFaceExtractor()
        self.arcface.initialize()

        self.age_predictor = MiVOLOPredictor()
        try:
            self.age_predictor.initialize()
        except Exception as e:
            logger.warning(f"年齡預測器初始化失敗，/analyze 將無法產生年齡: {e}")
            self.age_predictor = None

        # 下游分類器（base 兩條改為 10 折 GroupKFold 集成）
        self.lr_embedding = FoldEnsembleScorer(
            lr_embedding_folds_dir, name="LR#1(embedding, 10-fold)")
        self.lr_embedding.load()
        self.lr_asymmetry = FoldEnsembleScorer(
            lr_asymmetry_folds_dir, name="LR#2(asymmetry, 10-fold)")
        self.lr_asymmetry.load()
        self.tabpfn = TabPFNScorer(tabpfn_path, name="TabPFN(core3)")
        self.tabpfn.load()

        logger.info("=" * 60)
        logger.info("分析服務初始化完成")
        logger.info("=" * 60)

    def models_status(self) -> Dict[str, bool]:
        return {
            "arcface": self.arcface is not None and self.arcface._app is not None,
            "age_predictor": self.age_predictor is not None,
            "lr_embedding": bool(self.lr_embedding.models),
            "lr_asymmetry": bool(self.lr_asymmetry.models),
            "tabpfn": self.tabpfn.model is not None,
        }

    def analyze(self, images: List[np.ndarray], real_age: float) -> AnalysisResponse:
        start = time.time()
        logger.info("=" * 60)
        logger.info(f"開始分析（real_age={real_age}, 輸入 {len(images)} 張）")
        logger.info("=" * 60)

        try:
            # (2) 預處理
            with FacePreprocessor(self.config) as pp:
                faces = pp.process(images)
            if not faces:
                raise ValueError("預處理未產生有效結果（未偵測到正面人臉）")

            aligned = [f.aligned for f in faces]            # 全幀 → LR#1 embedding
            age_inputs = [f.age_crop for f in faces]        # 臉特寫 → MiVOLO 年齡
            lefts = [f.mirror_left for f in faces]          # 畫布原圖 → LR#2 左項
            rights = [f.mirror_right for f in faces]        # 畫布鏡射 → LR#2 右項

            # (3) 年齡（用 age_crop 臉特寫，對齊 analyze crop_faces.py → predict.py）
            if self.age_predictor is None:
                raise RuntimeError("年齡預測器不可用，無法計算 age_error")
            predicted_age = self.age_predictor.predict(age_inputs)
            if predicted_age is None:
                raise ValueError("年齡預測失敗（所有臉部皆無有效輸出）")
            age_error = float(real_age) - predicted_age
            logger.info(f"predicted_age={predicted_age:.2f}, age_error={age_error:.2f}")

            # (4.1) LR#1：aligned 全幀 embedding（訓練端 variant=original / OriginalSource）
            emb_aligned = [e for e in self.arcface.extract_batch(aligned) if e is not None]
            if not emb_aligned:
                raise ValueError("沒有有效的 aligned embedding（LR#1）")
            E_orig = np.array(emb_aligned, dtype=np.float64)  # (n, 512)
            embedding_LR_score = self.lr_embedding.score_mean(E_orig)

            # (4.2) LR#2：mirrors 畫布的 face_left / face_right embedding → 差 D = E_left − E_right
            emb_l = self.arcface.extract_batch(lefts)
            emb_r = self.arcface.extract_batch(rights)
            pairs = [(l, r) for l, r in zip(emb_l, emb_r) if l is not None and r is not None]
            if not pairs:
                raise ValueError("沒有有效的 face_left/face_right embedding 配對（LR#2）")
            E_left = np.array([p[0] for p in pairs], dtype=np.float64)   # (n, 512)
            E_right = np.array([p[1] for p in pairs], dtype=np.float64)  # (n, 512)
            logger.info(f"LR#1 用 {len(emb_aligned)} 張、LR#2 用 {len(pairs)} 組配對")

            D = E_left - E_right  # differences
            asymmetry_LR_score = self.lr_asymmetry.score_mean(D)
            logger.info(
                f"embedding_LR_score={embedding_LR_score:.4f}, "
                f"asymmetry_LR_score={asymmetry_LR_score:.4f}"
            )

            # (5) TabPFN(core3)
            ad_prob = self.tabpfn.predict(embedding_LR_score, asymmetry_LR_score, age_error)
            logger.info(f"ad_prob={ad_prob:.4f}")

            processing_time = time.time() - start
            logger.info(f"分析完成，耗時 {processing_time:.2f}s")

            return AnalysisResponse(
                success=True,
                error=None,
                predicted_age=round(predicted_age, 2),
                embedding_LR_score=embedding_LR_score,
                ad_prob=ad_prob,
                processing_time=processing_time,
                timestamp=datetime.now(),
            )

        except Exception as e:
            processing_time = time.time() - start
            logger.error(f"分析失敗: {e}")
            return AnalysisResponse(
                success=False,
                error=str(e),
                predicted_age=None,
                embedding_LR_score=None,
                ad_prob=None,
                processing_time=processing_time,
                timestamp=datetime.now(),
            )
