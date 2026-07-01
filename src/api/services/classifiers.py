"""
下游分類器載入與評分

對齊 Alz_face_analyze 訓練端契約：
  - LR#1 / LR#2（base）：10 折 GroupKFold base LR 集成。每條 = 10 個 sklearn
    Pipeline(StandardScaler + LogisticRegression, C=0.001, class_weight="balanced")，
    即產生 analyze forward OOF 的同一套折模型（export_deploy_folds.py 已逐格驗證
    重現落地 oof_scores.csv）。輸入 raw ArcFace 512-d；每折逐張 predict_proba[:,1]，
    對「所有(折 × 照片)」取平均 = 該 session 的 base 分數。
  - TabPFN：已在 core3 訓練表上 fit 的 TabPFNClassifier，以 pickle 序列化。
    core3 特徵順序 = [embedding_LR_score, asymmetry_LR_score, age_error]。
    註（範圍 A）：core3 表的 base 分數取自單折 OOF；此處推論改用 10 折平均，
    兩者分佈不完全一致（已與使用者確認接受，換取 base 分數可複現/可稽核）。
"""

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FoldEnsembleScorer:
    """10 折 base LR 集成（部署變體）。

    載入某條 base LR 的 K 個 fold Pipeline（每折含 StandardScaler + LogisticRegression），
    對每張照片用 K 個模型各打 predict_proba[:,1]，取所有 (折 × 照片) 的平均當作該
    session 的分數。那 K 個模型即 analyze forward OOF 的同一套折模型，故此分數可回頭
    以 workspace/deploy 的 fold 分數矩陣稽核。
    """

    def __init__(self, folds_dir: Path, name: str = "LR-ensemble"):
        self.folds_dir = Path(folds_dir)
        self.name = name
        self.models = []

    def load(self) -> None:
        import joblib

        if not self.folds_dir.is_dir():
            raise FileNotFoundError(f"找不到 {self.name} fold 目錄: {self.folds_dir}")
        paths = sorted(self.folds_dir.glob("fold_*.joblib"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(
                f"{self.name}: {self.folds_dir} 下沒有 fold_*.joblib")
        self.models = [joblib.load(p) for p in paths]
        logger.info(f"✓ {self.name} 載入 {len(self.models)} 個 fold 模型: {self.folds_dir.name}")

    def score_mean(self, features: np.ndarray) -> float:
        """features: (n_photos, 512) → 每折逐張 predict_proba[:,1] → 全部(折×張)取平均。"""
        if not self.models:
            raise RuntimeError(f"{self.name} 尚未 load()")
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"{self.name} 期望 2D 特徵，收到 shape={X.shape}")
        per_fold = np.vstack([m.predict_proba(X)[:, 1] for m in self.models])  # (K, n)
        return float(per_fold.mean())


class TabPFNScorer:
    """已在 core3 訓練表上 fit 的 TabPFNClassifier（pickle）。"""

    # core3 特徵順序，須與訓練端 META_FEATURE_SETS["core3"] 完全一致。
    FEATURE_ORDER = ("embedding_LR_score", "asymmetry_LR_score", "age_error")

    def __init__(self, model_path: Path, name: str = "TabPFN"):
        self.model_path = Path(model_path)
        self.name = name
        self.model = None

    def load(self) -> None:
        import torch

        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到 {self.name} 模型: {self.model_path}")

        if torch.cuda.is_available():
            # 有 GPU：pickle 內含的 cuda 張量可直接還原（與 fit 環境一致）。
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            # 無 GPU：此 pickle 在 device=auto→cuda 下 fit，torch 預設會拒絕把
            # cuda 張量反序列化到 CPU-only 機器。攔截 storage 還原並 map 到 cpu，
            # 載入後再 .to("cpu") 重設快取的 devices_ 並把內部 model/executor 搬到 CPU。
            import io

            orig_loader = torch.storage._load_from_bytes
            torch.storage._load_from_bytes = lambda b: torch.load(
                io.BytesIO(b), weights_only=False, map_location="cpu"
            )
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            finally:
                torch.storage._load_from_bytes = orig_loader
            if hasattr(self.model, "to"):
                self.model.to("cpu")
            logger.info(f"（無 GPU）{self.name} 已映射至 CPU")

        logger.info(f"✓ {self.name} 載入成功: {self.model_path.name}")

    def predict(self, embedding_lr_score: float, asymmetry_lr_score: float,
                age_error: float) -> float:
        """core3 三特徵 → AD 機率 predict_proba[:,1]。"""
        if self.model is None:
            raise RuntimeError(f"{self.name} 尚未 load()")
        vals = {
            "embedding_LR_score": embedding_lr_score,
            "asymmetry_LR_score": asymmetry_lr_score,
            "age_error": age_error,
        }
        X = np.array([[vals[k] for k in self.FEATURE_ORDER]], dtype=np.float64)
        return float(self.model.predict_proba(X)[0, 1])
