"""
src/api/routers/analyze.py
分析路由
"""

import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Annotated

from src.api.schemas import QuestionnaireData, AnalysisResponse, ErrorResponse
from src.api.services import AnalysisService, FileHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["analysis"])


def get_analysis_service() -> AnalysisService:
    """依賴注入：取得分析服務實例（由主程式提供）"""
    # 這個函數會在 app.py 中被覆寫
    raise NotImplementedError("AnalysisService 未設定")


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="人臉不對稱性與認知評估分析",
    description="上傳包含人臉照片的壓縮檔和問卷資料，回傳分析結果",
    responses={
        200: {
            "description": "分析成功",
            "model": AnalysisResponse
        },
        400: {
            "description": "請求錯誤（檔案格式、大小等）",
            "model": ErrorResponse
        },
        500: {
            "description": "伺服器錯誤",
            "model": ErrorResponse
        }
    }
)
async def analyze(
    file: Annotated[UploadFile, File(description="壓縮檔（支援 .zip, .7z, .rar）")],
    age: Annotated[int, Form(description="年齡（歲）", ge=0, le=150)],
    gender: Annotated[int, Form(description="性別 (0=女性, 1=男性)", ge=0, le=1)],
    education_years: Annotated[int, Form(description="教育年數", ge=0, le=30)],
    q1: Annotated[int, Form(description="問卷題目 1", ge=0, le=1)],
    q2: Annotated[int, Form(description="問卷題目 2", ge=0, le=2)],
    q3: Annotated[int, Form(description="問卷題目 3", ge=0, le=2)],
    q4: Annotated[int, Form(description="問卷題目 4", ge=0, le=1)],
    q5: Annotated[int, Form(description="問卷題目 5", ge=0, le=1)],
    q6: Annotated[int, Form(description="問卷題目 6", ge=0, le=1)],
    q7: Annotated[int, Form(description="問卷題目 7", ge=0, le=1)],
    q8: Annotated[int, Form(description="問卷題目 8", ge=0, le=1)],
    q9: Annotated[int, Form(description="問卷題目 9", ge=0, le=1)],
    q10: Annotated[int, Form(description="問卷題目 10", ge=0, le=1)],
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """
    執行人臉不對稱性與認知評估分析
    
    **輸入要求：**
    - 壓縮檔：包含 5-20 張正面人臉照片
    - 檔案大小：≤ 500MB
    - 支援格式：.zip, .7z, .rar
    - 圖片格式：JPG, JPEG, PNG, BMP, TIFF
    
    **問卷資料：**
    - age: 年齡
    - gender: 性別 (0=女性, 1=男性)
    - education_years: 教育年數
    - q1-q10: 問卷題目回答
    
    **回傳結果：**
    - q6ds_result: 6QDS 認知評估分數 (0.0-1.0)
    - marked_figure: Base64 編碼的標記人臉圖片
    - processing_time: 處理時間（秒）
    """
    logger.info(f"收到分析請求: {file.filename}")
    
    # 驗證檔案格式
    supported_formats = FileHandler.get_supported_formats()
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_formats:
        error_msg = f"不支援的檔案格式: {file_ext}，支援格式: {', '.join(supported_formats)}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # 驗證檔案大小
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    file_content = await file.read()
    
    if len(file_content) > MAX_FILE_SIZE:
        error_msg = f"檔案大小超過限制（{len(file_content) / 1024 / 1024:.1f}MB > 500MB）"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # 建立問卷資料
    try:
        questionnaire = QuestionnaireData(
            age=age,
            gender=gender,
            education_years=education_years,
            q1=q1, q2=q2, q3=q3, q4=q4, q5=q5,
            q6=q6, q7=q7, q8=q8, q9=q9, q10=q10
        )
    except Exception as e:
        error_msg = f"問卷資料驗證失敗: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # 儲存臨時檔案
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(file_content)
        temp_file.flush()
        temp_file.close()
        
        temp_path = Path(temp_file.name)
        
        logger.info(f"開始分析: {file.filename}")
        
        # 執行分析
        result = service.analyze(temp_path, questionnaire)
        
        if not result.success:
            logger.error(f"分析失敗: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        
        logger.info(f"分析完成: {file.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"分析過程發生錯誤: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # 清理臨時檔案
        if temp_file and Path(temp_file.name).exists():
            try:
                Path(temp_file.name).unlink()
            except Exception as e:
                logger.warning(f"清理臨時檔案失敗: {e}")