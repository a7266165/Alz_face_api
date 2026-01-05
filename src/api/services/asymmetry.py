"""
src/api/services/asymmetry.py
人臉不對稱性分析服務
"""

import logging
import json
import re
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import xgboost as xgb

from src.core import FacePreprocessor, FeatureExtractor, APIConfig, MiVOLOPredictor

logger = logging.getLogger(__name__)


class AsymmetryAnalyzer:
    """人臉不對稱性分析器"""
    
    # 年齡校正閾值
    AGE_CORRECTION_THRESHOLD = 65.0
    
    def __init__(
        self,
        classifier_path: Path,
        feature_selection_paths: Dict[str, Path],  # ← 字典
        n_select: int = 10
    ):
        """
        初始化分析器
        
        Args:
            classifier_path: XGBoost 分類器路徑
            feature_selection_path: 特徵選取資訊路徑
            n_select: 選擇最正面的圖片數量
        """
        self.classifier_path = Path(classifier_path)
        self.feature_selection_paths = {k: Path(v) for k, v in feature_selection_paths.items()}
        self.feature_selections = {}
        self.n_select = n_select
        
        # 初始化元件
        self.config = APIConfig(n_select=n_select, save_intermediate=False)
        self.preprocessor = None
        self.feature_extractor = None
        self.classifier = None
        self.feature_selection = None
        self.age_predictor = None
        
        self._load_models()
    
    def _load_models(self):
        """載入所有模型"""
        # 載入特徵提取器
        try:
            self.feature_extractor = FeatureExtractor()
            if not self.feature_extractor.available_models:
                raise RuntimeError("沒有可用的特徵提取模型")
            
            logger.info("✓ 特徵提取器初始化完成")
        except Exception as e:
            raise RuntimeError(f"特徵提取器初始化失敗: {e}")
        
        # 載入特徵選取資訊
        try:
            for feature_type, path in self.feature_selection_paths.items():
                with open(path, 'r') as f:
                    self.feature_selections[feature_type] = json.load(f)
                logger.info(f"✓ 特徵選取載入 ({feature_type}): {self.feature_selections[feature_type]['selected_dim']} 維")
        except Exception as e:
            raise RuntimeError(f"載入特徵選取失敗: {e}")
        
        # 載入分類器
        try:
            self.classifier = xgb.Booster()
            self.classifier.load_model(str(self.classifier_path))
            logger.info(f"✓ 不對稱性分類器載入: {self.classifier_path.name}")
        except Exception as e:
            raise RuntimeError(f"載入分類器失敗: {e}")
        
        # 載入年齡預測器
        try:
            self.age_predictor = MiVOLOPredictor()
            self.age_predictor.initialize()
            logger.info("✓ 年齡預測器初始化完成")
        except Exception as e:
            logger.warning(f"年齡預測器初始化失敗，將跳過年齡校正: {e}")
            self.age_predictor = None
    
    def _sigmoid_correction(self, age: float) -> float:
        """
        計算年齡校正係數
        
        Args:
            age: 預測年齡
            
        Returns:
            校正係數 (0 ~ 0.3)
        """
        # y = 0.3 / (1 + exp(-0.1 * (age - 32.5)))
        return 0.3 / (1 + math.exp(-0.1 * (age - 32.5)))
    
    def analyze(self, images: List[np.ndarray]) -> Optional[float]:
        """
        分析人臉不對稱性
        
        Args:
            images: 圖片列表（BGR 格式）
            
        Returns:
            預測分數 (0.0-1.0)，失敗則回傳 None
        """
        try:
            # Step 1: 預處理
            logger.info("執行預處理...")
            with FacePreprocessor(self.config) as preprocessor:
                processed_faces = preprocessor.process(images)
            
            if not processed_faces:
                logger.warning("預處理未產生有效結果")
                return None
            
            logger.info(f"預處理完成: {len(processed_faces)} 張")
            
            # Step 2: 預測年齡
            predicted_age = None
            if self.age_predictor:
                logger.info("預測年齡...")
                aligned_images = [face.aligned for face in processed_faces if face.aligned is not None]
                if aligned_images:
                    predicted_age = self.age_predictor.predict(aligned_images)
                    if predicted_age:
                        logger.info(f"預測年齡: {predicted_age:.1f} 歲")
            
            # Step 3: 提取特徵
            logger.info("提取 TopoFR 特徵...")
            left_images = [face.left_mirror for face in processed_faces]
            right_images = [face.right_mirror for face in processed_faces]
            
            left_features_dict = self.feature_extractor.extract_features(
                left_images, models=["topofr"], verbose=False
            )
            right_features_dict = self.feature_extractor.extract_features(
                right_images, models=["topofr"], verbose=False
            )
            
            left_features = left_features_dict["topofr"]
            right_features = right_features_dict["topofr"]
            
            # 過濾 None
            valid_pairs = [
                (l, r) for l, r in zip(left_features, right_features)
                if l is not None and r is not None
            ]
            
            if not valid_pairs:
                logger.warning("沒有有效的特徵對")
                return None
            
            left_array = np.array([p[0] for p in valid_pairs])
            right_array = np.array([p[1] for p in valid_pairs])
            
            logger.info(f"提取完成: {len(valid_pairs)} 對特徵")
            
            # Step 4: 計算組合特徵
            logger.info("計算組合特徵...")
            combined_features = self._compute_combined_features(left_array, right_array)
            logger.info(f"組合特徵: {combined_features.shape}")
            
            # Step 5: 預測
            logger.info("執行預測...")
            dmatrix = xgb.DMatrix(combined_features.reshape(1, -1))
            prediction = float(self.classifier.predict(dmatrix)[0])
            
            # Step 6: 年齡校正
            if predicted_age is not None and predicted_age < self.AGE_CORRECTION_THRESHOLD:
                correction_factor = self._sigmoid_correction(predicted_age)
                corrected_prediction = prediction * correction_factor
                logger.info(
                    f"年齡校正: {prediction:.4f} × {correction_factor:.4f} = {corrected_prediction:.4f}"
                )
                prediction = corrected_prediction
            
            logger.info(f"✓ 不對稱性預測: {prediction:.4f}")
            return prediction
            
        except Exception as e:
            logger.error(f"不對稱性分析失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_combined_features(
            self, 
            left_array: np.ndarray, 
            right_array: np.ndarray
        ) -> np.ndarray:
            """
            計算組合特徵：average + absolute_relative_differences
            
            Args:
                left_array: 左臉特徵 (N, 512)
                right_array: 右臉特徵 (N, 512)
                
            Returns:
                組合特徵向量 (selected_dim_avg + selected_dim_abs_rel,)
            """
            # 計算 average: (L + R) / 2
            avg_features = (left_array + right_array) / 2
            avg_features = np.mean(avg_features, axis=0)  # (512,)
            
            # 計算 absolute_relative_differences: |L - R| / √(L² + R²)
            abs_diff = np.abs(left_array - right_array)
            norm = np.sqrt(left_array**2 + right_array**2)
            abs_rel_features = np.zeros_like(abs_diff)
            mask = norm > 1e-8
            abs_rel_features[mask] = abs_diff[mask] / norm[mask]
            abs_rel_features = np.mean(abs_rel_features, axis=0)  # (512,)
            
            # 特徵篩選
            avg_indices = self.feature_selections["average"]["selected_indices"]
            abs_rel_indices = self.feature_selections["absolute_relative_differences"]["selected_indices"]
            
            avg_selected = avg_features[avg_indices]
            abs_rel_selected = abs_rel_features[abs_rel_indices]
            
            # 串接
            combined = np.concatenate([avg_selected, abs_rel_selected])
            
            return combined