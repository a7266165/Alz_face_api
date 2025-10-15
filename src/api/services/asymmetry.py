"""
src/api/services/asymmetry.py
人臉不對稱性分析服務
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import xgboost as xgb

from src.core import FacePreprocessor, FeatureExtractor, APIConfig

logger = logging.getLogger(__name__)


class AsymmetryAnalyzer:
    """人臉不對稱性分析器"""
    
    def __init__(
        self,
        classifier_path: Path,
        feature_selection_path: Path,
        n_select: int = 10
    ):
        """
        初始化分析器
        
        Args:
            classifier_path: XGBoost 分類器路徑（例如: arcface_difference_cdr0.5.json）
            feature_selection_path: 特徵選取資訊路徑
            n_select: 選擇最正面的圖片數量
        """
        self.classifier_path = Path(classifier_path)
        self.feature_selection_path = Path(feature_selection_path)
        self.n_select = n_select
        
        # 從檔名解析模型名稱和特徵類型
        self.model_name, self.feature_type = self._parse_classifier_name()
        
        # 初始化元件
        self.config = APIConfig(n_select=n_select, save_intermediate=False)
        self.preprocessor = None
        self.feature_extractor = None
        self.classifier = None
        self.feature_selection = None
        
        self._load_models()
    
    def _parse_classifier_name(self) -> Tuple[str, str]:
        """
        從分類器檔名解析模型名稱和特徵類型
        
        例如: arcface_difference_cdr0.5.json -> ("arcface", "difference")
        
        Returns:
            (model_name, feature_type)
        """
        filename = self.classifier_path.stem  # 去掉副檔名
        
        # 使用正則表達式解析
        # 格式: {model}_{feature_type}_{cdr_info}
        match = re.match(r'([^_]+)_([^_]+)_cdr', filename)
        
        if not match:
            raise ValueError(
                f"無法解析分類器檔名: {filename}\n"
                f"預期格式: {{model}}_{{feature_type}}_cdr{{threshold}}.json"
            )
        
        model_name = match.group(1)
        feature_type = match.group(2)
        
        # 驗證
        valid_models = {'arcface', 'dlib', 'topofr'}
        valid_types = {'difference', 'average', 'relative'}
        
        if model_name not in valid_models:
            raise ValueError(f"不支援的模型: {model_name}，可用: {valid_models}")
        
        if feature_type not in valid_types:
            raise ValueError(f"不支援的特徵類型: {feature_type}，可用: {valid_types}")
        
        logger.info(f"分類器配置: 模型={model_name}, 特徵類型={feature_type}")
        return model_name, feature_type
    
    def _load_models(self):
        """載入所有模型"""
        # 載入預處理器
        try:
            self.preprocessor = FacePreprocessor(self.config)
            logger.info("✓ 預處理器初始化完成")
        except Exception as e:
            raise RuntimeError(f"預處理器初始化失敗: {e}")
        
        # 載入特徵提取器
        try:
            self.feature_extractor = FeatureExtractor()
            if not self.feature_extractor.available_models:
                raise RuntimeError("沒有可用的特徵提取模型")
            
            # 檢查所需模型是否可用
            if self.model_name not in self.feature_extractor.available_models:
                raise RuntimeError(
                    f"模型 {self.model_name} 不可用，"
                    f"可用模型: {self.feature_extractor.available_models}"
                )
            
            logger.info("✓ 特徵提取器初始化完成")
        except Exception as e:
            raise RuntimeError(f"特徵提取器初始化失敗: {e}")
        
        # 載入特徵選取資訊
        try:
            with open(self.feature_selection_path, 'r') as f:
                self.feature_selection = json.load(f)
            logger.info(
                f"✓ 特徵選取載入: "
                f"{self.feature_selection['original_dim']} → "
                f"{self.feature_selection['selected_dim']} 維"
            )
        except Exception as e:
            raise RuntimeError(f"載入特徵選取失敗: {e}")
        
        # 載入分類器
        try:
            self.classifier = xgb.Booster()
            self.classifier.load_model(str(self.classifier_path))
            logger.info(f"✓ 不對稱性分類器載入: {self.classifier_path.name}")
        except Exception as e:
            raise RuntimeError(f"載入分類器失敗: {e}")
    
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
            with self.preprocessor:
                processed_faces = self.preprocessor.process(images)
            
            if not processed_faces:
                logger.warning("預處理未產生有效結果")
                return None
            
            logger.info(f"預處理完成: {len(processed_faces)} 對左右臉")
            
            # Step 2: 收集左右臉鏡射圖片
            left_images = [face.left_mirror for face in processed_faces]
            right_images = [face.right_mirror for face in processed_faces]
            
            # Step 3: 提取特徵（只提取指定模型）
            logger.info(f"提取 {self.model_name} 特徵...")
            left_features_dict = self.feature_extractor.extract_features(
                left_images,
                models=[self.model_name],
                verbose=False
            )
            right_features_dict = self.feature_extractor.extract_features(
                right_images,
                models=[self.model_name],
                verbose=False
            )
            
            # 取出特徵列表
            left_features = left_features_dict[self.model_name]
            right_features = right_features_dict[self.model_name]
            
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
            
            # Step 4: 計算差異特徵
            logger.info(f"計算 {self.feature_type} 特徵...")
            diff_features = self._calculate_differences(left_array, right_array)
            
            # Step 5: 對多張圖片的特徵取平均
            feature_vector = np.mean(diff_features, axis=0)
            logger.info(f"特徵向量: {feature_vector.shape}")
            
            # Step 6: 特徵選取
            logger.info("執行特徵選取...")
            selected_features = self._select_features(feature_vector)
            logger.info(f"選取後: {selected_features.shape}")
            
            # Step 7: 預測
            logger.info("執行預測...")
            prediction = self._predict(selected_features)
            
            logger.info(f"✓ 不對稱性預測: {prediction:.4f}")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"不對稱性分析失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_differences(
        self,
        left_array: np.ndarray,
        right_array: np.ndarray
    ) -> np.ndarray:
        """
        計算左右臉差異特徵
        
        Args:
            left_array: 左臉特徵陣列 (N, D)
            right_array: 右臉特徵陣列 (N, D)
            
        Returns:
            差異特徵陣列 (N, D)
        """
        if self.feature_type == "difference":
            # 差異: left - right
            return left_array - right_array
        
        elif self.feature_type == "average":
            # 平均: (left + right) / 2
            return (left_array + right_array) / 2
        
        elif self.feature_type == "relative":
            # 相對差異: abs(left - right) / abs(left + right)
            abs_diff = np.abs(left_array - right_array)
            abs_sum = np.abs(left_array + right_array)
            
            # 避免除以零
            relative = np.zeros_like(abs_diff)
            mask = abs_sum > 1e-8
            relative[mask] = abs_diff[mask] / abs_sum[mask]
            
            return relative
        
        else:
            raise ValueError(f"未知的特徵類型: {self.feature_type}")
    
    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """
        根據 feature_selection.json 選取特徵
        
        Args:
            features: 完整特徵向量
            
        Returns:
            選取後的特徵向量
        """
        selected_indices = self.feature_selection['selected_indices']
        return features[selected_indices]
    
    def _predict(self, features: np.ndarray) -> float:
        """
        使用分類器預測
        
        Args:
            features: 選取後的特徵向量
            
        Returns:
            預測分數
        """
        # 建立 DMatrix
        dmatrix = xgb.DMatrix(features.reshape(1, -1))
        
        # 預測
        prediction = self.classifier.predict(dmatrix)[0]
        
        return prediction