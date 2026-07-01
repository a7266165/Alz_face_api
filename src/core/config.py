"""
核心預處理 / 管線配置

與 Alz_face_analyze 訓練端對齊的參數（鏡射法、bg_mode、選圖數…）。
推論端與訓練端必須一致，否則特徵分布漂移會讓 LR / TabPFN 失準。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

from .mediapipe_utils import MIDLINE_POINTS


def _default_landmark_service_url() -> str:
    return os.environ.get("LANDMARK_SERVICE_URL", "http://127.0.0.1:8771/landmarks")


@dataclass
class MirrorConfig:
    """鏡射生成配置（對齊 Alz_face_analyze/src/config.py:MirrorConfig）。"""

    # "flip"：對齊後整張水平翻轉（original + cv2.flip）；訓練端部署採用此法。
    # "midline"：沿臉部中線 PCA 精確半臉鏡射（保留以備切換）。
    mirror_method: str = "flip"
    mirror_size: Tuple[int, int] = (512, 512)  # 輸出鏡射 / 原圖畫布大小
    feather_px: int = 2                        # midline 法邊緣羽化（flip 不用）
    margin: float = 0.08                       # 畫布邊緣留白比例
    midline_points: Tuple[int, ...] = MIDLINE_POINTS


@dataclass
class PipelineConfig:
    """臉部預處理管線配置。"""

    # ---- 選圖 ----
    # 偵測信心度非本設定所控——實際 min_detection_confidence 硬編在 landmark 子服務
    # （landmark_service/server.py，numpy<2）。
    n_select: int = 10                  # 選最正面的張數

    # ---- Landmark 偵測服務（mp.solutions，numpy<2 子服務）----
    # 選圖/對齊/裁切的 landmark 皆打這個服務（與訓練端同一偵測器）。
    landmark_service_url: str = field(default_factory=_default_landmark_service_url)
    landmark_service_timeout: float = 30.0

    # ---- 年齡分支臉部裁切（對齊 analyze scripts/age/crop_faces.py）----
    # 在對齊圖上重偵測 landmark，取所有點外接框外擴此比例（以框長邊為基準）後裁切，
    # 供 MiVOLO 年齡預測。訓練端預設 0.35。僅影響年齡分支，不動 ArcFace/LR。
    age_crop_margin: float = 0.35

    # ---- 去背 ----
    # "background"：保留背景（部署採用，與訓練 bg_mode 一致）→ 不跑 apply_mask。
    # "no_background"：臉部凸包以外塗黑。
    bg_mode: str = "background"

    # ---- 鏡射 ----
    mirror: MirrorConfig = field(default_factory=MirrorConfig)

    @property
    def apply_mask(self) -> bool:
        """是否做凸包去背。"""
        return self.bg_mode == "no_background"
