"""
src/api/routers/health.py
健康檢查與資訊路由
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from typing import Dict

from src.api.schemas import HealthResponse
from src.api.services import AnalysisService, FileHandler

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# API 版本
API_VERSION = "1.0.0"


def get_analysis_service() -> AnalysisService:
    """依賴注入：取得分析服務實例（由主程式提供）"""
    raise NotImplementedError("AnalysisService 未設定")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康檢查",
    description="檢查 API 服務狀態和模型載入情況"
)
async def health_check(
    service: AnalysisService = Depends(get_analysis_service)
) -> HealthResponse:
    """
    健康檢查端點
    
    **回傳資訊：**
    - status: 服務狀態（healthy/unhealthy）
    - version: API 版本
    - models_loaded: 各模型載入狀態
    - timestamp: 檢查時間
    """
    try:
        # 檢查各個模型是否載入成功
        models_status = {
            "6qds_predictor": service.q6ds_predictor.model is not None,
            "feature_extractor": len(service.asymmetry_analyzer.feature_extractor.available_models) > 0,
            "classifier": service.asymmetry_analyzer.classifier is not None,
            "visualizer": service.visualizer.face_mesh is not None,
        }
        
        # 額外資訊
        models_status["available_feature_models"] = service.asymmetry_analyzer.feature_extractor.available_models
        models_status["classifier_config"] = {
            "model": service.asymmetry_analyzer.model_name,
            "feature_type": service.asymmetry_analyzer.feature_type
        }
        
        # 判斷整體狀態
        all_healthy = all(models_status.values()[:4])  # 檢查前4個核心模型
        status = "healthy" if all_healthy else "unhealthy"
        
        logger.info(f"健康檢查: {status}")
        
        return HealthResponse(
            status=status,
            version=API_VERSION,
            models_loaded=models_status,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            version=API_VERSION,
            models_loaded={"error": str(e)},
            timestamp=datetime.now()
        )


@router.get(
    "/",
    summary="API 資訊",
    description="取得 API 基本資訊和使用說明"
)
async def root() -> Dict:
    """
    API 根路徑
    
    **回傳資訊：**
    - API 名稱和版本
    - 可用端點
    - 支援的檔案格式
    - 使用說明連結
    """
    return {
        "name": "人臉不對稱性與認知評估 API",
        "version": API_VERSION,
        "description": "上傳人臉照片壓縮檔和問卷資料，回傳 6QDS 認知評估和不對稱性分析結果",
        "endpoints": {
            "analyze": {
                "method": "POST",
                "path": "/analyze",
                "description": "執行人臉分析和認知評估"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "健康檢查"
            },
            "docs": {
                "method": "GET",
                "path": "/docs",
                "description": "Swagger 互動式文檔"
            },
            "redoc": {
                "method": "GET",
                "path": "/redoc",
                "description": "ReDoc 文檔"
            }
        },
        "supported_formats": {
            "archive": FileHandler.get_supported_formats(),
            "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        },
        "requirements": {
            "file_size_limit": "500MB",
            "recommended_images": "5-20 張正面人臉照片",
            "questionnaire_fields": [
                "age (年齡)",
                "gender (性別: 0=女性, 1=男性)",
                "education_years (教育年數)",
                "q1-q10 (問卷題目 1-10)"
            ]
        },
        "contact": {
            "repository": "https://github.com/a7266165/Alz_face_api"
        }
    }