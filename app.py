"""
app.py
FastAPI 主程式 - 人臉不對稱性與認知評估 API
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# 加入專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.services import AnalysisService
from src.api.routers import api_router, analyze, health
from src.api.middleware import (
    logging_middleware,
    error_handler_middleware,
    setup_exception_handlers,
    setup_logging
)

logger = logging.getLogger(__name__)

# ==================== 配置 ====================

class Config:
    """應用程式配置"""
    
    # API 資訊
    API_TITLE = "人臉不對稱性與認知評估 API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = """
    ## 功能
    上傳人臉照片壓縮檔和問卷資料，進行：
    - 6QDS 認知評估
    - 人臉不對稱性分析
    - 人臉標記視覺化
    
    ## 輸入要求
    - **壓縮檔**: 包含 5-20 張正面人臉照片（支援 .zip, .7z, .rar）
    - **檔案大小**: ≤ 50MB
    - **圖片格式**: JPG, JPEG, PNG, BMP, TIFF
    - **問卷資料**: 年齡、性別、教育年數、10 題問卷回答
    
    ## 回傳結果
    - `q6ds_result`: 6QDS 認知評估分數 (0.0-1.0)
    - `marked_figure`: Base64 編碼的標記人臉圖片
    - `processing_time`: 處理時間（秒）
    """
    
    # 模型檔案路徑
    MODEL_DIR = project_root / "model"
    Q6DS_MODEL = MODEL_DIR / "xgb_6qds_model.json"
    CLASSIFIER_MODEL = MODEL_DIR / "xgb_classifier.json"
    FEATURE_SELECTION = MODEL_DIR / "feature_selection.json"
    
    # 分析參數
    N_SELECT = 10  # 選擇最正面的圖片數量
    
    # 日誌配置
    LOG_LEVEL = "INFO"
    
    # CORS 配置
    ALLOW_ORIGINS = ["*"]  # 生產環境應限制來源
    ALLOW_METHODS = ["GET", "POST"]
    ALLOW_HEADERS = ["*"]


# ==================== 全域變數 ====================

# 分析服務實例（啟動時初始化）
analysis_service: AnalysisService = None


# ==================== 生命週期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程式生命週期管理
    
    啟動時：載入模型和服務
    關閉時：清理資源
    """
    global analysis_service
    
    logger.info("=" * 70)
    logger.info("🚀 啟動人臉分析與認知評估 API")
    logger.info("=" * 70)
    
    # 檢查模型檔案
    _check_model_files()
    
    # 初始化分析服務
    try:
        logger.info("載入模型和服務...")
        analysis_service = AnalysisService(
            q6ds_model_path=Config.Q6DS_MODEL,
            classifier_path=Config.CLASSIFIER_MODEL,
            feature_selection_path=Config.FEATURE_SELECTION,
            n_select=Config.N_SELECT
        )
        logger.info("✓ 服務初始化完成")
    except Exception as e:
        logger.error(f"✗ 服務初始化失敗: {e}")
        raise
    
    logger.info("=" * 70)
    logger.info(f"API 文檔: http://localhost:8000/docs")
    logger.info(f"健康檢查: http://localhost:8000/health")
    logger.info("=" * 70)
    
    yield
    
    # 關閉時清理
    logger.info("關閉 API 服務...")


def _check_model_files():
    """檢查必要的模型檔案是否存在"""
    required_files = {
        "6QDS 模型": Config.Q6DS_MODEL,
        "分類器模型": Config.CLASSIFIER_MODEL,
        "特徵選取": Config.FEATURE_SELECTION,
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
            logger.warning(f"⚠️  模型檔案不存在: {path}")
        else:
            logger.info(f"✓ {name}: {path.name}")
    
    if missing:
        logger.warning(
            f"缺少 {len(missing)} 個模型檔案\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


# ==================== 依賴注入 ====================

def get_analysis_service() -> AnalysisService:
    """
    取得分析服務實例（依賴注入）
    
    這個函數會覆寫 routers 中的同名函數
    """
    if analysis_service is None:
        raise RuntimeError("AnalysisService 尚未初始化")
    return analysis_service


# ==================== FastAPI 應用 ====================

# 設定日誌
setup_logging(log_level=Config.LOG_LEVEL)

# 建立 FastAPI 應用
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=Config.ALLOW_METHODS,
    allow_headers=Config.ALLOW_HEADERS,
)

# 註冊中介軟體（順序很重要：先日誌，後錯誤處理）
app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=error_handler_middleware)

# 設定異常處理器
setup_exception_handlers(app)

# 覆寫路由中的依賴注入
app.dependency_overrides[analyze.get_analysis_service] = get_analysis_service
app.dependency_overrides[health.get_analysis_service] = get_analysis_service

# 註冊路由
app.include_router(api_router)

# ==================== 主程式入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開發模式：自動重載
        log_level=Config.LOG_LEVEL.lower()
    )