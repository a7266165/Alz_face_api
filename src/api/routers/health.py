"""
健康檢查路由：/health 與 /
"""

import logging

from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse
from src.api.services import AnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])

API_VERSION = "2.0.0"


def get_analysis_service() -> AnalysisService:
    """依賴注入：取得分析服務（在 app.py 被覆寫）。"""
    raise NotImplementedError("AnalysisService 未設定")


@router.get("/health", response_model=HealthResponse, summary="健康檢查")
async def health(service: AnalysisService = Depends(get_analysis_service)) -> HealthResponse:
    """回傳服務狀態與各模型載入情形（永遠 200，由 status 欄位判斷）。"""
    try:
        models = service.models_status()
        status_str = "healthy" if all(models.values()) else "degraded"
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        models = {}
        status_str = "unhealthy"

    return HealthResponse(status=status_str, version=API_VERSION, models_loaded=models)


@router.get("/", summary="API 資訊")
async def root() -> dict:
    return {
        "name": "人臉失智相關評估 API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {"analyze": "POST /analyze", "health": "GET /health"},
    }
