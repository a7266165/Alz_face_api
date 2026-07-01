"""
分析路由：/analyze

輸入（multipart/form-data）：
  - file: 單一上傳檔——.mp4，或壓縮檔 .zip/.7z/.rar（內含 n 張 .jpg/.jpeg/.png）
  - real_age: 真實年齡（歲）

輸出：AnalysisResponse { predicted_age, embedding_LR_score, ad_prob }
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.schemas import AnalysisResponse, ErrorResponse
from src.api.services import AnalysisService, FileHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["analysis"])

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def get_analysis_service() -> AnalysisService:
    """依賴注入：取得分析服務（在 app.py 被覆寫）。"""
    raise NotImplementedError("AnalysisService 未設定")


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="人臉失智相關評估",
    description=(
        "上傳單一檔案（.mp4 或壓縮檔 .zip/.7z/.rar，內含 n 張 .jpg/.png）與真實年齡，"
        "回傳 predicted_age / embedding_LR_score / ad_prob"
    ),
    responses={
        200: {"description": "分析成功", "model": AnalysisResponse},
        400: {"description": "請求錯誤", "model": ErrorResponse},
        500: {"description": "伺服器錯誤", "model": ErrorResponse},
    },
)
async def analyze(
    file: Annotated[
        UploadFile,
        File(description="單一 .mp4，或壓縮檔 .zip/.7z/.rar（內含 n 張 .jpg/.jpeg/.png）"),
    ],
    real_age: Annotated[float, Form(description="真實年齡（歲）", ge=0, le=150)],
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="未收到任何檔案")

    logger.info(f"收到分析請求: {file.filename}, real_age={real_age}")

    # 讀取內容並檢查大小
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"檔案大小超過限制（{len(data) / 1024 / 1024:.1f}MB > "
            f"{MAX_FILE_SIZE // 1024 // 1024}MB）",
        )

    # 解碼成 BGR 影像列表：.mp4 抽幀 / 壓縮檔解壓讀圖（格式、解壓失敗皆丟 ValueError → 400）
    try:
        images = FileHandler.load_single(file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 執行分析（下游 pipeline 與輸入型別無關）
    result = service.analyze(images, real_age)
    if not result.success:
        # 業務失敗（偵測不到臉 / 年齡或 embedding 失敗）→ 500
        raise HTTPException(status_code=500, detail=result.error)
    return result
