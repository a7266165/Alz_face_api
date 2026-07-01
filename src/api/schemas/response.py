"""API 回應資料模型。"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    """/analyze 結果（輸出三個分數）。"""

    success: bool = Field(..., description="分析是否成功")
    error: Optional[str] = Field(None, description="錯誤訊息（失敗時）")

    # (3) MiVOLO 年齡預測（10 張平均）
    predicted_age: Optional[float] = Field(None, description="預測年齡（歲）")

    # (4.1) LR#1：原圖 ArcFace embedding 逐張分數取平均
    embedding_LR_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="embedding LR 分數 (0-1)")

    # (5) TabPFN(core3) AD 機率
    ad_prob: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="TabPFN 失智相關機率 (0-1)")

    processing_time: Optional[float] = Field(None, description="處理時間（秒）")
    timestamp: Optional[datetime] = Field(None, description="分析時間")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "error": None,
                "predicted_age": 71.32,
                "embedding_LR_score": 0.6428,
                "ad_prob": 0.7135,
                "processing_time": 12.4,
                "timestamp": "2026-06-22T12:34:56",
            }
        }


class ErrorResponse(BaseModel):
    """錯誤回應（統一格式，供 middleware / exception handler 使用）。"""

    success: bool = Field(False, description="永遠是 False")
    error: str = Field(..., description="錯誤訊息")
    error_type: Optional[str] = Field(None, description="錯誤類型")
    details: Optional[dict] = Field(None, description="詳細錯誤資訊")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """健康檢查回應。"""

    status: str = Field(..., description="服務狀態 (healthy/unhealthy)")
    version: str = Field(..., description="API 版本")
    models_loaded: dict = Field(..., description="已載入的模型狀態")
    timestamp: datetime = Field(default_factory=datetime.now)
