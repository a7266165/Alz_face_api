"""
src/core/feature_extract.py
API 和 Analyze 共用的特徵提取模組
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import dlib

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    深度學習特徵提取器
    
    支援模型：
    - Dlib: 128 維特徵
    - ArcFace: 512 維特徵 (via InsightFace)
    - TopoFR: 512 維特徵
    """
    
    def __init__(self):
        """初始化特徵提取器"""
        self.available_models = []
        
        # 使用 external 目錄作為模型根目錄
        self.external_dir = Path(__file__).parent.parent.parent / "external"
        
        # 初始化所有模型
        self._init_dlib()
        self._init_arcface()
        self._init_topofr()
        
        # 報告狀態
        self._report_status()
    
    def _report_status(self):
        """報告模型載入狀態"""
        if self.available_models:
            logger.info("=" * 50)
            logger.info("特徵提取器初始化完成")
            logger.info(f"可用模型: {', '.join(self.available_models)}")
            for model in self.available_models:
                dim = self.get_feature_dim(model)
                logger.info(f"  - {model}: {dim} 維")
            logger.info("=" * 50)
        else:
            logger.warning("警告：沒有任何特徵提取模型可用")
            logger.warning("請安裝至少一個: pip install dlib / insightface")
    
    # ========== 模型初始化 ==========
    
    def _init_dlib(self):
        """初始化 Dlib (128維)"""
        try:
            import dlib
        except ImportError:
            logger.debug("Dlib 未安裝，跳過")
            return
        
        try:
            # 人臉檢測器
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            # 檢查模型檔案
            dlib_dir = self.external_dir / "dlib"
            predictor_path = dlib_dir / "shape_predictor_68_face_landmarks.dat"
            face_rec_path = dlib_dir / "dlib_face_recognition_resnet_model_v1.dat"
            
            missing_files = []
            if not predictor_path.exists():
                missing_files.append(str(predictor_path))
            if not face_rec_path.exists():
                missing_files.append(str(face_rec_path))
            
            if missing_files:
                logger.warning(
                    f"Dlib 模型檔案缺失:\n" + 
                    "\n".join(f"  - {f}" for f in missing_files) +
                    "\n請下載並放置到 external/dlib/ 目錄"
                )
                logger.info("下載連結: http://dlib.net/files/")
                return
            
            # 載入模型
            self.dlib_predictor = dlib.shape_predictor(str(predictor_path))
            self.dlib_face_rec = dlib.face_recognition_model_v1(str(face_rec_path))
            
            self.available_models.append("dlib")
            logger.info("✓ Dlib 載入成功")
            
        except Exception as e:
            logger.warning(f"Dlib 初始化失敗: {e}")
    
    def _init_arcface(self):
        """初始化 ArcFace (512維)"""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.debug("InsightFace 未安裝，跳過 ArcFace")
            return
        
        try:
            # 使用預設設定（會自動下載模型）
            self.arcface_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.arcface_app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.available_models.append("arcface")
            logger.info("✓ ArcFace 載入成功")
            
        except Exception as e:
            logger.warning(f"ArcFace 初始化失敗: {e}")
    
    def _init_topofr(self):
        """初始化 TopoFR (512維)"""
        try:
            import torch
        except ImportError:
            logger.debug("PyTorch 未安裝，跳過 TopoFR")
            return
        
        try:
            # 嘗試載入 TopoFR
            topofr_path = self.external_dir / "TopoFR"
            if topofr_path.exists():
                sys.path.insert(0, str(topofr_path))
                from backbones import get_model
            else:
                logger.debug("TopoFR 路徑不存在，跳過")
                return
            
            # 尋找模型
            model_dir = topofr_path / "model"
            model_files = list(model_dir.glob("*TopoFR*.pt")) + \
                         list(model_dir.glob("*TopoFR*.pth"))
            
            if not model_files:
                logger.warning(f"TopoFR 模型檔案不存在於: {model_dir}")
                return
            
            model_path = model_files[0]
            
            # 判斷架構
            model_name = model_path.stem.upper()
            if "R100" in model_name:
                network = "r100"
            elif "R50" in model_name:
                network = "r50"
            elif "R200" in model_name:
                network = "r200"
            else:
                network = "r100"
            
            # 載入
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.topofr_model = get_model(network, fp16=False)
            
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get('state_dict', checkpoint)
            
            self.topofr_model.load_state_dict(checkpoint, strict=False)
            self.topofr_model.to(device).eval()
            self.topofr_device = device
            
            self.available_models.append("topofr")
            logger.info(f"✓ TopoFR 載入成功 ({network}, 設備: {device})")
            
        except Exception as e:
            logger.warning(f"TopoFR 初始化失敗: {e}")
    
    # ========== 特徵提取 ==========
    
    def extract_features(
        self,
        images: List[np.ndarray],
        models: Union[str, List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """
        批次提取特徵
        
        Args:
            images: 影像列表（例如：20張左右臉鏡射）
            models: 模型名稱或列表
                - None: 使用所有可用模型
                - "dlib": 單一模型
                - ["dlib", "arcface"]: 多個模型
            verbose: 是否顯示進度
            
        Returns:
            {model_name: [features1, features2, ...]}
        """
        # 統一轉成列表
        if models is None:
            models = self.available_models
        elif isinstance(models, str):
            models = [models]
        
        # 驗證模型名稱
        invalid = set(models) - set(self.available_models)
        if invalid:
            logger.warning(f"以下模型不可用，將跳過: {invalid}")
        
        valid_models = [m for m in models if m in self.available_models]
        
        if not valid_models:
            logger.error("沒有可用的模型")
            return {}
        
        # 模型提取方法映射
        extractors = {
            "dlib": self._extract_dlib,
            "arcface": self._extract_arcface,
            "topofr": self._extract_topofr
        }
        
        # 提取特徵
        results = {model: [] for model in valid_models}
        
        for i, image in enumerate(images):
            if verbose and i % 10 == 0:
                logger.info(f"處理進度: {i}/{len(images)}")
            
            for model in valid_models:
                try:
                    features = extractors[model](image)
                    results[model].append(features)
                except Exception as e:
                    logger.error(f"{model} 提取失敗: {e}")
                    results[model].append(None)
        
        # 統計
        for model, features in results.items():
            success = sum(1 for f in features if f is not None)
            logger.info(f"{model}: {success}/{len(images)} 成功")
        
        return results

    def _extract_dlib(self, image: np.ndarray) -> Optional[np.ndarray]:
        """內部方法：提取 Dlib 特徵 (128維)"""
        import cv2
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray, 1)
        
        if not faces:
            h, w = gray.shape[:2]
            faces = [dlib.rectangle(0, 0, w, h)]
            logger.debug("Dlib 未檢測到人臉，使用整張圖")
        
        shape = self.dlib_predictor(gray, faces[0])
        descriptor = self.dlib_face_rec.compute_face_descriptor(image, shape)
        
        return np.array(descriptor, dtype=np.float32)

    def _extract_arcface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """內部方法：提取 ArcFace 特徵 (512維)"""
        import cv2
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.arcface_app.get(image_rgb)
        
        if not faces:
            # 偵測失敗時，直接用 recognition model 處理整張圖
            logger.debug("ArcFace 未檢測到人臉，使用整張圖")
            
            # 縮放到模型輸入尺寸並提取特徵
            img_resized = cv2.resize(image_rgb, (112, 112))
            img_input = np.transpose(img_resized, (2, 0, 1))[np.newaxis, ...]
            img_input = (img_input - 127.5) / 127.5
            img_input = img_input.astype(np.float32)
            embedding = self.arcface_app.models['recognition'].forward(img_input)
            return embedding.flatten().astype(np.float32)
        
        return faces[0].embedding.astype(np.float32)

    def _extract_topofr(self, image: np.ndarray) -> Optional[np.ndarray]:
        """內部方法：提取 TopoFR 特徵 (512維)"""
        import torch
        import cv2
        
        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.div(255).sub(0.5).div(0.5).to(self.topofr_device)
        
        with torch.no_grad():
            embedding = self.topofr_model(img)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0].astype(np.float32)
    
    # ========== 差異計算 ==========
    
    def calculate_differences(self, left_features, right_features, methods=None):
        if methods is None:
            raise ValueError("必須明確指定 methods 參數")
        
        valid_methods = {"differences", "absolute_differences", "averages", "relative_differences", "absolute_relative_differences"}
        if invalid := set(methods) - valid_methods:
            raise ValueError(f"未知的方法: {invalid}")
        
        results = {}
        diff = None
        
        if "differences" in methods:
            diff = left_features - right_features
            results["embedding_differences"] = diff.astype(np.float32)

        if "absolute_differences" in methods:
            abs_diff = np.abs(left_features - right_features)
            results["embedding_absolute_differences"] = abs_diff.astype(np.float32)

        if "averages" in methods:
            results["embedding_averages"] = ((left_features + right_features) / 2).astype(np.float32)
        
        if "relative_differences" in methods:
            diff = left_features - right_features
            norm = np.sqrt(left_features**2 + right_features**2)

            rel_diff = np.zeros_like(diff)
            mask = norm > 1e-8
            rel_diff[mask] = diff[mask] / norm[mask]
            
            results["embedding_relative_differences"] = rel_diff.astype(np.float32)

        if "absolute_relative_differences" in methods:
            abs_diff = np.abs(left_features - right_features)
            norm = np.sqrt(left_features**2 + right_features**2)

            rel_abs_diff = np.zeros_like(abs_diff)
            mask = norm > 1e-8
            rel_abs_diff[mask] = abs_diff[mask] / norm[mask]
            
            results["embedding_absolute_relative_differences"] = rel_abs_diff.astype(np.float32)
        return results
    # ========== 人口學特徵 ==========
    
    def add_demographics(
        self,
        features_list: List[np.ndarray],
        ages: List[float],
        genders: List[float]
    ) -> List[np.ndarray]:
        """
        加入人口學特徵
        
        Args:
            features_list: 特徵向量列表
            ages: 年齡列表
            genders: 性別列表 (0=女, 1=男)
            
        Returns:
            結合後的特徵向量列表
        """
        if not (len(features_list) == len(ages) == len(genders)):
            raise ValueError(
                f"長度不一致: features={len(features_list)}, "
                f"ages={len(ages)}, genders={len(genders)}"
            )
        
        return [
            np.concatenate([feat, np.array([age, gender], dtype=np.float32)])
            for feat, age, gender in zip(features_list, ages, genders)
        ]
    
    # ========== 工具方法 ==========
    
    def get_available_models(self) -> List[str]:
        """取得可用的模型列表"""
        return self.available_models.copy()
    
    def get_feature_dim(self, model_name: str) -> Optional[int]:
        """
        取得模型輸出維度
        
        Returns:
            特徵維度，不存在返回 None
        """
        dim_map = {
            "dlib": 128,
            "arcface": 512,
            "topofr": 512
        }
        return dim_map.get(model_name)
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        驗證特徵向量有效性
        
        Args:
            features: 特徵向量
            
        Returns:
            是否有效
        """
        if features is None:
            return False
        
        # 檢查維度
        if features.ndim != 1:
            logger.warning(f"特徵維度錯誤: {features.shape}")
            return False
        
        # 檢查數值
        if not np.isfinite(features).all():
            logger.warning("特徵包含 NaN 或 Inf")
            return False
        
        # 檢查是否全零
        if np.allclose(features, 0):
            logger.warning("特徵向量全為零")
            return False
        
        return True
    
    def get_status_report(self) -> Dict[str, any]:
        """
        取得狀態報告
        
        Returns:
            包含模型狀態的字典
        """
        report = {
            "available_models": self.available_models,
            "model_dimensions": {
                model: self.get_feature_dim(model)
                for model in self.available_models
            },
            "has_gpu": False
        }
        
        # 檢查 GPU
        try:
            import torch
            report["has_gpu"] = torch.cuda.is_available()
            if report["has_gpu"]:
                report["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        return report