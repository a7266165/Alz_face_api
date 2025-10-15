"""
src/api/middleware/logging.py
請求/回應日誌中介軟體
"""

import logging
import time
from fastapi import Request

logger = logging.getLogger(__name__)


async def logging_middleware(request: Request, call_next):
    """
    記錄每個請求的基本資訊和處理時間
    
    Args:
        request: FastAPI 請求物件
        call_next: 下一個中介軟體或路由處理函數
    
    Returns:
        回應物件
    """
    # 開始時間
    start_time = time.time()
    
    # 客戶端資訊
    client_host = request.client.host if request.client else "unknown"
    
    # 記錄請求開始
    logger.info(
        f"→ {request.method} {request.url.path} "
        f"from {client_host}"
    )
    
    # 執行請求
    try:
        response = await call_next(request)
        
        # 計算處理時間
        process_time = time.time() - start_time
        
        # 記錄請求完成
        logger.info(
            f"← {request.method} {request.url.path} "
            f"[{response.status_code}] "
            f"in {process_time:.3f}s"
        )
        
        # 將處理時間加入回應標頭
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        return response
    
    except Exception as exc:
        # 計算處理時間（即使發生錯誤）
        process_time = time.time() - start_time
        
        # 記錄請求失敗
        logger.error(
            f"✗ {request.method} {request.url.path} "
            f"failed in {process_time:.3f}s: {type(exc).__name__}: {str(exc)}"
        )
        
        # 重新拋出異常，讓錯誤處理中介軟體處理
        raise


def setup_logging(log_level: str = "INFO", log_format: str = None):
    """
    設定應用程式日誌
    
    Args:
        log_level: 日誌等級（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_format: 自訂日誌格式（None 使用預設）
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 降低第三方套件的日誌等級
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger.info("✓ 日誌系統設定完成")