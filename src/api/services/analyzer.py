"""
src/api/services/analyzer.py
主分析服務：整合三條處理流程
"""

import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.api.schemas import QuestionnaireData, AnalysisResponse, ErrorResponse
from src.api.services.file_handler import FileHandler
from src.api.services.predictor import Q6DSPredictor
from src.api.services.visualizer import FaceVisualizer
from src.api.services.asymmetry import AsymmetryAnalyzer

logger = logging.getLogger(__name__)


class AnalysisService:
    """主分析服務"""
    
    def __init__(
        self,
        q6ds_model_path: Path,
        classifier_path: Path,
        feature_selection_path: Path,
        n_select: int = 10
    ):
        """
        初始化分析服務
        
        Args:
            q6ds_model_path: 6QDS 模型路徑
            classifier_path: 不對稱性分類器路徑
            feature_selection_path: 特徵選取資訊路徑
            n_select: 選擇最正面的圖片數量
        """
        self.file_handler = FileHandler()
        self.q6ds_predictor = Q6DSPredictor(q6ds_model_path)
        self.visualizer = FaceVisualizer()
        self.asymmetry_analyzer = AsymmetryAnalyzer(
            classifier_path,
            feature_selection_path,
            n_select
        )
        
        logger.info("=" * 60)
        logger.info("分析服務初始化完成")
        logger.info("=" * 60)
    
    def analyze(
        self,
        archive_path: Path,
        questionnaire: QuestionnaireData
    ) -> AnalysisResponse:
        """
        執行完整分析
        
        Args:
            archive_path: 壓縮檔路徑
            questionnaire: 問卷資料
            
        Returns:
            分析結果
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("開始分析")
        logger.info("=" * 60)
        
        try:
            # Step 1: 解壓縮並載入圖片
            logger.info("\n[1/4] 處理壓縮檔...")
            temp_dir, temp_path = self.file_handler.create_temp_dir()
            
            try:
                image_dir = self.file_handler.extract_archive(archive_path, temp_path)
                images = self.file_handler.load_images(image_dir)
                logger.info(f"✓ 載入 {len(images)} 張圖片")
                
                # Step 2: Flow 1 - 6QDS 預測
                logger.info("\n[2/4] 6QDS 認知評估...")
                q6ds_result = self.q6ds_predictor.predict(questionnaire)
                logger.info(f"✓ 6QDS 預測: {q6ds_result:.4f}")
                
                # Step 3: Flow 2 - 不對稱性分析（最耗時）
                logger.info("\n[3/4] 人臉不對稱性分析...")
                # asymmetry_result = self.asymmetry_analyzer.analyze(images)
                # 暫時不執行不對稱性分析（因為模型可能還沒訓練好）
                asymmetry_result = None
                
                # Step 4: Flow 3 - 生成標記圖片
                logger.info("\n[4/4] 生成標記圖片...")
                marked_figure = self.visualizer.generate_marked_image(images[0])
                
                # 計算處理時間
                processing_time = time.time() - start_time
                
                # 整合結果
                logger.info("=" * 60)
                logger.info("分析完成")
                logger.info(f"總處理時間: {processing_time:.2f} 秒")
                logger.info("=" * 60)
                
                return AnalysisResponse(
                    success=True,
                    error=None,
                    q6ds_result=q6ds_result,
                    marked_figure=marked_figure,
                    processing_time=processing_time,
                    timestamp=datetime.now()
                )
                
            finally:
                # 清理臨時目錄
                temp_dir.cleanup()
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error("=" * 60)
            logger.error(f"分析失敗: {error_msg}")
            logger.error("=" * 60)
            
            return AnalysisResponse(
                success=False,
                error=error_msg,
                q6ds_result=None,
                marked_figure=None,
                processing_time=processing_time,
                timestamp=datetime.now()
            )