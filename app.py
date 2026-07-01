"""
app.py
FastAPI 主程式 — 人臉失智相關評估 API（ArcFace + LR×2 + TabPFN）
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# 加入專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core import PipelineConfig
from src.api.services import AnalysisService
from src.api.routers import api_router, analyze, health
from src.api.middleware import (
    logging_middleware,
    setup_exception_handlers,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

class Config:
    """應用程式配置。"""

    API_TITLE = "人臉失智相關評估 API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = """
    ## 功能
    上傳人臉影片或多張照片 + 真實年齡，回傳：
    - `predicted_age`: MiVOLO 預測年齡（10 張平均）
    - `embedding_LR_score`: 原圖 ArcFace embedding 的 base LR 分數（10 折集成平均）
    - `ad_prob`: TabPFN(core3) 失智相關機率

    ## 輸入
    - **檔案**: 單一 `.mp4`，或壓縮檔 `.zip/.7z/.rar`（內含 n 張 `.jpg/.jpeg/.png`）；`file` 欄位單檔
    - **real_age**: 真實年齡（歲）
    - **檔案大小**: ≤ 500MB

    ## Pipeline
    preprocess(選 10 張最正面 → 對齊 → flip 鏡射) → MiVOLO 年齡 →
    ArcFace embedding(20 張) → LR#1 / LR#2(L−R 差，各 10 折集成) → TabPFN(core3)
    """

    # 模型檔案（部署時放入 model/）
    MODEL_DIR = project_root / "model"
    # base 兩條 LR 改為 10 折 GroupKFold 集成：各一個目錄，內含 fold_0..9.joblib
    LR_EMBEDDING_FOLDS = MODEL_DIR / "embedding"     # LR#1（variant=original）10 折
    LR_ASYMMETRY_FOLDS = MODEL_DIR / "asymmetry"    # LR#2（variant=differences）10 折
    TABPFN_MODEL = MODEL_DIR / "tabpfn_core3.pkl"             # TabPFN（feature_set=core3）

    # 管線參數（須與訓練端一致）
    N_SELECT = 10
    BG_MODE = "background"          # 不去背
    MIRROR_METHOD = "flip"         # 對齊後整張水平翻轉

    LOG_LEVEL = "INFO"

    # CORS（生產環境應限制來源）
    ALLOW_ORIGINS = ["*"]
    ALLOW_METHODS = ["GET", "POST"]
    ALLOW_HEADERS = ["*"]


# ==================== 全域 ====================

analysis_service: AnalysisService = None


# ==================== 生命週期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analysis_service

    logger.info("=" * 70)
    logger.info("🚀 啟動人臉失智相關評估 API")
    logger.info("=" * 70)

    _check_model_files()

    try:
        logger.info("載入模型和服務...")
        config = PipelineConfig(n_select=Config.N_SELECT, bg_mode=Config.BG_MODE)
        config.mirror.mirror_method = Config.MIRROR_METHOD
        analysis_service = AnalysisService(
            lr_embedding_folds_dir=Config.LR_EMBEDDING_FOLDS,
            lr_asymmetry_folds_dir=Config.LR_ASYMMETRY_FOLDS,
            tabpfn_path=Config.TABPFN_MODEL,
            config=config,
        )
        logger.info("✓ 服務初始化完成")
    except Exception as e:
        logger.error(f"✗ 服務初始化失敗: {e}")
        raise

    logger.info("=" * 70)
    logger.info("API 文檔: http://localhost:8000/docs")
    logger.info("健康檢查: http://localhost:8000/health")
    logger.info("=" * 70)

    yield

    logger.info("關閉 API 服務...")


def _check_model_files():
    """檢查部署模型是否就位（缺檔僅警告，初始化時才真正報錯）。"""
    missing = []
    fold_dirs = {
        "LR#1 (embedding, 10-fold)": Config.LR_EMBEDDING_FOLDS,
        "LR#2 (asymmetry, 10-fold)": Config.LR_ASYMMETRY_FOLDS,
    }
    for name, d in fold_dirs.items():
        n = len(list(d.glob("fold_*.joblib"))) if d.is_dir() else 0
        if n:
            logger.info(f"✓ {name}: {n} 折 @ model/{d.relative_to(Config.MODEL_DIR).as_posix()}")
        else:
            missing.append(f"{name}: {d}")
            logger.warning(f"⚠️  fold 模型不存在: {d}")

    if Config.TABPFN_MODEL.exists():
        logger.info(f"✓ TabPFN (core3): {Config.TABPFN_MODEL.name}")
    else:
        missing.append(f"TabPFN (core3): {Config.TABPFN_MODEL}")
        logger.warning(f"⚠️  模型檔案不存在: {Config.TABPFN_MODEL}")

    if missing:
        logger.warning(f"缺少 {len(missing)} 個模型:\n" + "\n".join(f"  - {m}" for m in missing))


# ==================== 依賴注入 ====================

def get_analysis_service() -> AnalysisService:
    if analysis_service is None:
        raise RuntimeError("AnalysisService 尚未初始化")
    return analysis_service


# ==================== FastAPI ====================

setup_logging(log_level=Config.LOG_LEVEL)

app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=Config.ALLOW_METHODS,
    allow_headers=Config.ALLOW_HEADERS,
)

# 中介軟體：logging（計時 + X-Process-Time header）。未攔截例外的統一 500
# 改由 setup_exception_handlers 的 @app.exception_handler(Exception) 於最外層處理。
app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)

setup_exception_handlers(app)

# 覆寫路由 DI
app.dependency_overrides[analyze.get_analysis_service] = get_analysis_service
app.dependency_overrides[health.get_analysis_service] = get_analysis_service

app.include_router(api_router)


# ==================== 主程式 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=Config.LOG_LEVEL.lower(),
    )
