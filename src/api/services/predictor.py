"""
src/api/services/predictor.py
6QDS 認知評估預測服務
"""

import logging
from pathlib import Path
import numpy as np
import xgboost as xgb

from src.api.schemas import QuestionnaireData

logger = logging.getLogger(__name__)


class Q6DSPredictor:
    """6QDS 認知評估預測器"""
    
    def __init__(self, model_path: Path):
        """
        初始化預測器
        
        Args:
            model_path: XGBoost 模型路徑
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """載入 XGBoost 模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到 6QDS 模型: {self.model_path}")
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            logger.info(f"✓ 6QDS 模型載入成功: {self.model_path.name}")
        except Exception as e:
            raise RuntimeError(f"載入 6QDS 模型失敗: {e}")
    
    def predict(self, questionnaire: QuestionnaireData) -> float:
        """
        預測 6QDS 分數
        
        Args:
            questionnaire: 問卷資料
            
        Returns:
            預測分數 (0.0-1.0)
        """
        if self.model is None:
            raise RuntimeError("6QDS 模型未載入")
        
        try:
            # 轉換為特徵陣列
            features = questionnaire.to_feature_array()
            
            # 建立 DMatrix
            feature_names = [
                '年紀', '性別', '教育程度yr',
                'q1', 'q2', 'q3', 'q4', 'q5',
                'q6', 'q7', 'q8', 'q9', 'q10'
            ]
            dmatrix = xgb.DMatrix(
                np.array(features).reshape(1, -1),
                feature_names=feature_names
            )
            
            # 預測
            prediction = self.model.predict(dmatrix)[0]
            
            logger.debug(f"6QDS 預測結果: {prediction:.4f}")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"6QDS 預測失敗: {e}")
            raise RuntimeError(f"6QDS 預測失敗: {e}")