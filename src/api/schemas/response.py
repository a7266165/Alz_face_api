"""
API 回應資料模型
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AnalysisResponse(BaseModel):
    """分析結果回應"""
    
    success: bool = Field(
        ...,
        description="分析是否成功"
    )
    
    error: Optional[str] = Field(
        None,
        description="錯誤訊息（如果失敗）"
    )
    
    # 6QDS 認知評估結果
    q6ds_result: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="6QDS 認知評估預測分數 (0.0-1.0)"
    )
    
    # 標記圖片
    marked_figure: Optional[str] = Field(
        None,
        description="Base64 編碼的標記人臉圖片"
    )
    
    # 元資料（可選）
    processing_time: Optional[float] = Field(
        None,
        description="處理時間（秒）"
    )
    
    timestamp: Optional[datetime] = Field(
        None,
        description="分析時間"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "error": None,
                "q6ds_result": 0.75,
                "marked_figure": "data:image/jpeg;base64,/9j/4AAQ...",
                "processing_time": 15.3,
                "timestamp": "2025-10-14T12:34:56"
            }
        }


class ErrorResponse(BaseModel):
    """錯誤回應（統一格式）"""
    
    success: bool = Field(
        False,
        description="永遠是 False"
    )
    
    error: str = Field(
        ...,
        description="錯誤訊息"
    )
    
    error_type: Optional[str] = Field(
        None,
        description="錯誤類型（ValidationError, FileError, SystemError 等）"
    )
    
    details: Optional[dict] = Field(
        None,
        description="詳細錯誤資訊"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="錯誤發生時間"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "無法偵測到人臉",
                "error_type": "ProcessingError",
                "details": {
                    "uploaded_images": 10,
                    "valid_faces": 0
                },
                "timestamp": "2025-10-14T12:34:56"
            }
        }


class HealthResponse(BaseModel):
    """健康檢查回應"""
    
    status: str = Field(
        ...,
        description="服務狀態 (healthy/unhealthy)"
    )
    
    version: str = Field(
        ...,
        description="API 版本"
    )
    
    models_loaded: dict = Field(
        ...,
        description="已載入的模型狀態"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="檢查時間"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {
                    "6qds": True
                },
                "timestamp": "2025-10-14T12:34:56"
            }
        }