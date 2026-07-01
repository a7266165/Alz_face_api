"""
請求/回應日誌中介軟體
"""

import logging
import time
from fastapi import Request

logger = logging.getLogger(__name__)


async def logging_middleware(request: Request, call_next):
    """記錄每個請求的基本資訊與處理時間，並寫入 X-Process-Time header。"""
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"→ {request.method} {request.url.path} from {client_host}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"← {request.method} {request.url.path} "
            f"[{response.status_code}] in {process_time:.3f}s"
        )
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} "
            f"failed in {process_time:.3f}s: {type(exc).__name__}: {exc}"
        )
        raise


def setup_logging(log_level: str = "INFO", log_format: str = None):
    """設定應用程式日誌。"""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("✓ 日誌系統設定完成")
