"""
統一錯誤處理中介軟體
"""

import logging
from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def setup_exception_handlers(app):
    """設定特定異常處理器（含未攔截例外的 catch-all 500）。"""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        logger.warning(f"請求驗證失敗: {request.url.path} | {errors}")
        error_messages = [
            f"{' -> '.join(str(x) for x in e['loc'])}: {e['msg']}" for e in errors
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "請求資料驗證失敗",
                "error_type": "ValidationError",
                "details": {"errors": error_messages},
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        logger.warning(f"Pydantic 驗證失敗: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "資料驗證失敗",
                "error_type": "ValidationError",
                "details": {"errors": exc.errors()},
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.error(f"值錯誤: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(exc),
                "error_type": "ValueError",
                "details": None,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.error(f"檔案未找到: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "伺服器配置錯誤：缺少必要檔案",
                "error_type": "FileNotFoundError",
                "details": {"message": str(exc)},
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.error(f"執行時錯誤: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "執行時錯誤",
                "error_type": "RuntimeError",
                "details": {"message": str(exc)},
                "timestamp": datetime.now().isoformat(),
            },
        )

    # catch-all：任何未被上面攔截的例外 → 統一 500。取代原 error_handler_middleware，
    # 由 Starlette ServerErrorMiddleware 於最外層處理，payload 與行為等價、少一層 ASGI wrapper。
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"未處理的異常: {type(exc).__name__}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "伺服器內部錯誤",
                "error_type": type(exc).__name__,
                "details": {"message": str(exc), "path": str(request.url.path)},
                "timestamp": datetime.now().isoformat(),
            },
        )

    logger.info("✓ 異常處理器設定完成")
