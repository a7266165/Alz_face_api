"""
src/api/middleware/error_handler.py
統一錯誤處理中介軟體
"""

import logging
import traceback
from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """
    全域錯誤處理中介軟體
    
    捕獲所有未處理的異常，轉換為統一的錯誤回應格式
    """
    try:
        response = await call_next(request)
        return response
    
    except Exception as exc:
        # 記錄詳細錯誤資訊
        logger.error(
            f"未處理的異常: {type(exc).__name__}: {str(exc)}",
            exc_info=True
        )
        
        # 回傳統一格式的錯誤回應
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "伺服器內部錯誤",
                "error_type": type(exc).__name__,
                "details": {
                    "message": str(exc),
                    "path": str(request.url.path)
                },
                "timestamp": datetime.now().isoformat()
            }
        )


def setup_exception_handlers(app):
    """
    設定特定異常處理器
    
    Args:
        app: FastAPI 應用實例
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """處理請求驗證錯誤（Pydantic）"""
        errors = exc.errors()
        
        logger.warning(
            f"請求驗證失敗: {request.url.path}\n"
            f"錯誤: {errors}"
        )
        
        # 格式化錯誤訊息
        error_messages = []
        for error in errors:
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            error_messages.append(f"{field}: {message}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "請求資料驗證失敗",
                "error_type": "ValidationError",
                "details": {
                    "errors": error_messages,
                    "raw_errors": errors
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """處理 Pydantic 驗證錯誤"""
        logger.warning(f"Pydantic 驗證失敗: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "資料驗證失敗",
                "error_type": "ValidationError",
                "details": {
                    "errors": exc.errors()
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """處理值錯誤（通常是業務邏輯錯誤）"""
        logger.error(f"值錯誤: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(exc),
                "error_type": "ValueError",
                "details": None,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """處理檔案未找到錯誤"""
        logger.error(f"檔案未找到: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "伺服器配置錯誤：缺少必要檔案",
                "error_type": "FileNotFoundError",
                "details": {
                    "message": str(exc)
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        """處理執行時錯誤"""
        logger.error(f"執行時錯誤: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "執行時錯誤",
                "error_type": "RuntimeError",
                "details": {
                    "message": str(exc)
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    logger.info("✓ 異常處理器設定完成")