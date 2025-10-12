"""
核心預處理模組
統一的臉部預處理實作，供 API 和 Analyze 共用
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import shutil

from .config import PreprocessConfig

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """單張臉部資訊"""

    image: np.ndarray
    angle: float  # 中軸角度（度）
    confidence: float  # 偵測信心度
    landmarks: np.ndarray  # 468個特徵點座標
    path: Optional[Path] = None  # 原始檔案路徑
    index: int = 0  # 在批次中的索引


@dataclass
class ProcessedFace:
    """處理後的臉部資料"""

    left_mirror: np.ndarray  # 左臉鏡射
    right_mirror: np.ndarray  # 右臉鏡射
    original: Optional[np.ndarray] = None  # 原始影像
    aligned: Optional[np.ndarray] = None  # 對齊後影像
    metadata: Dict[str, Any] = None  # 元資料


class FacePreprocessor:
    """統一的臉部預處理器"""

    # MediaPipe 臉部中軸線定義（從 legacy 移植）
    FACEMESH_MID_LINE = [
        (10, 151),
        (151, 9),
        (9, 8),
        (8, 168),
        (168, 6),
        (6, 197),
        (197, 195),
        (195, 5),
        (5, 4),
        (4, 1),
        (1, 19),
        (19, 94),
        (94, 2),
    ]

    def __init__(self, config: PreprocessConfig):
        """
        初始化預處理器

        Args:
            config: 預處理配置
        """
        self.config = config

        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.detection_confidence,
        )

        # 設定工作區
        self._setup_workspace()

    def __enter__(self):
        """Context manager 進入"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 離開，釋放資源"""
        if self.face_mesh:
            self.face_mesh.close()

    def _setup_workspace(self):
        """設定工作區目錄結構"""
        if self.config.save_intermediate and self.config.workspace_dir:
            # 建立必要的子目錄
            subdirs = ["selected", "aligned", "mirrors", "debug"]
            for subdir in subdirs:
                path = self.config.workspace_dir / subdir
                path.mkdir(parents=True, exist_ok=True)

            logger.info(f"工作區設定完成: {self.config.workspace_dir}")

    def cleanup_workspace(self):
        """清理工作區（主要給 API 使用）"""
        if self.config.workspace_dir and self.config.workspace_dir.exists():
            shutil.rmtree(self.config.workspace_dir)
            logger.info(f"已清理工作區: {self.config.workspace_dir}")

    # ========== 主要處理流程 ==========

    def process(
        self, images: List[np.ndarray], image_paths: Optional[List[Path]] = None
    ) -> List[ProcessedFace]:
        """
        完整預處理流程

        Args:
            images: 輸入影像列表
            image_paths: 對應的檔案路徑（可選，用於命名）

        Returns:
            處理後的臉部資料列表
        """
        if not images:
            logger.warning("沒有輸入影像")
            return []

        logger.info(f"開始處理 {len(images)} 張影像")

        # Step 1: 分析所有影像，提取臉部資訊
        face_infos = self._analyze_all_faces(images, image_paths)
        logger.info(f"成功偵測 {len(face_infos)} 張臉部")

        if not face_infos:
            logger.warning("沒有偵測到任何臉部")
            return []

        # Step 2: 選擇最正面的 n 張
        if "select" in self.config.steps:
            selected = self._select_best_faces(face_infos)
            logger.info(f"選擇最正面的 {len(selected)} 張")

            if self.config.save_intermediate:
                self._save_selected(selected)
        else:
            selected = face_infos

        # Step 3: 處理選中的影像
        processed_faces = []
        for i, face_info in enumerate(selected):
            try:
                processed = self._process_single_face(face_info, i)
                processed_faces.append(processed)
            except Exception as e:
                logger.error(f"處理第 {i} 張臉部時失敗: {e}")
                continue

        logger.info(f"完成處理，共 {len(processed_faces)} 張成功")
        return processed_faces

    def _process_single_face(self, face_info: FaceInfo, index: int) -> ProcessedFace:
        """
        處理單張臉部

        Args:
            face_info: 臉部資訊
            index: 處理索引

        Returns:
            處理後的臉部資料
        """
        current_image = face_info.image
        aligned_image = None

        # Step 1: 角度校正
        if "align" in self.config.steps and self.config.align_face:
            aligned_image = self._align_face(current_image, face_info.angle)
            current_image = aligned_image

            if self.config.save_intermediate:
                self._save_aligned(aligned_image, face_info, index)

        # Step 2: 生成鏡射
        if "mirror" in self.config.steps:
            # 重新偵測對齊後影像的特徵點
            if aligned_image is not None:
                # 重新偵測特徵點
                rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    landmarks = self._landmarks_to_array(
                        results.multi_face_landmarks[0], current_image.shape
                    )
                else:
                    landmarks = face_info.landmarks
            else:
                landmarks = face_info.landmarks

            left_mirror, right_mirror = self._create_mirror_images(
                current_image, landmarks
            )
        else:
            # 如果不做鏡射，左右都使用原圖
            left_mirror = current_image.copy()
            right_mirror = current_image.copy()

        # Step 3: CLAHE 增強
        if "clahe" in self.config.steps and self.config.apply_clahe:
            left_mirror = self._apply_clahe(left_mirror)
            right_mirror = self._apply_clahe(right_mirror)

        # 儲存鏡射結果
        if self.config.save_intermediate:
            self._save_mirrors(left_mirror, right_mirror, face_info, index)

        # 封裝結果
        return ProcessedFace(
            left_mirror=left_mirror,
            right_mirror=right_mirror,
            original=face_info.image if self.config.save_intermediate else None,
            aligned=aligned_image,
            metadata={
                "angle": float(face_info.angle),
                "confidence": float(face_info.confidence),
                "path": str(face_info.path) if face_info.path else None,
                "index": index,
            },
        )

    # ========== 臉部分析與選擇 ==========

    def _analyze_all_faces(
        self, images: List[np.ndarray], paths: Optional[List[Path]] = None
    ) -> List[FaceInfo]:
        """
        分析所有影像，提取臉部資訊

        Args:
            images: 影像列表
            paths: 對應路徑列表

        Returns:
            臉部資訊列表
        """
        face_infos = []

        for i, image in enumerate(images):
            # 轉換色彩空間
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                logger.debug(f"第 {i} 張影像未偵測到臉部")
                continue

            # 提取特徵點
            landmarks = results.multi_face_landmarks[0]
            landmarks_array = self._landmarks_to_array(landmarks, image.shape)

            # 計算中軸角度
            angle = self._calculate_midline_angle(landmarks, image.shape)

            face_infos.append(
                FaceInfo(
                    image=image,
                    angle=angle,
                    confidence=1.0,  # MediaPipe 不直接提供信心度
                    landmarks=landmarks_array,
                    path=paths[i] if paths else None,
                    index=i,
                )
            )

        return face_infos

    def _select_best_faces(self, face_infos: List[FaceInfo]) -> List[FaceInfo]:
        """
        選擇最正面的 n 張臉

        Args:
            face_infos: 所有臉部資訊

        Returns:
            選中的臉部資訊
        """
        # 按角度絕對值排序
        sorted_faces = sorted(face_infos, key=lambda x: abs(x.angle))

        # 選擇前 n 張
        n_select = min(self.config.n_select, len(sorted_faces))
        selected = sorted_faces[:n_select]

        # 記錄選擇結果
        for face in selected:
            logger.debug(f"選擇: 索引={face.index}, 角度={face.angle:.2f}°")

        return selected

    def _calculate_midline_angle(
        self, landmarks, image_shape: Tuple[int, int]
    ) -> float:
        """
        計算臉部中軸線角度

        Args:
            landmarks: MediaPipe 特徵點
            image_shape: 影像尺寸

        Returns:
            角度（度）
        """
        h, w = image_shape[:2]
        angles = []

        for pair in self.FACEMESH_MID_LINE:
            point1 = landmarks.landmark[pair[0]]
            point2 = landmarks.landmark[pair[1]]

            # 轉換為像素座標
            x1, y1 = point1.x * w, point1.y * h
            x2, y2 = point2.x * w, point2.y * h

            # 計算向量角度
            dx = x2 - x1
            dy = y2 - y1

            if dy != 0:
                angle = np.arctan(dx / dy)
                angles.append(np.degrees(angle))
            else:
                # 垂直線段
                angles.append(90.0 if dx > 0 else -90.0)

        # 返回平均角度
        return np.mean(angles) if angles else 0.0

    # ========== 影像處理函數 ==========

    def _align_face(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋轉影像使臉部垂直

        Args:
            image: 輸入影像
            angle: 旋轉角度（度）

        Returns:
            旋轉後的影像
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 建立旋轉矩陣
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # 執行旋轉
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def _create_mirror_images(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成左右臉鏡射（完整版本，從 Analyze legacy 移植）

        Args:
            image: 輸入影像
            landmarks: 468個特徵點

        Returns:
            (左臉鏡射, 右臉鏡射)
        """
        # 建立臉部遮罩
        mask = self._build_face_mask(image.shape, landmarks)

        # 估計中線
        p0, n = self._estimate_midline(landmarks)

        # 生成左右鏡射
        left_mirror = self._align_to_canvas_premul(
            image,
            mask,
            p0,
            n,
            side="left",
            out_size=self.config.mirror_size,
            margin=self.config.margin,
        )

        right_mirror = self._align_to_canvas_premul(
            image,
            mask,
            p0,
            n,
            side="right",
            out_size=self.config.mirror_size,
            margin=self.config.margin,
        )

        return left_mirror, right_mirror

    def _build_face_mask(
        self, img_shape: Tuple[int, int], face_points: np.ndarray
    ) -> np.ndarray:
        """
        建立臉部遮罩

        Args:
            img_shape: 影像尺寸
            face_points: 臉部特徵點

        Returns:
            遮罩影像
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        if face_points.shape[0] == 0:
            return mask

        # 計算凸包
        hull = cv2.convexHull(face_points.astype(np.int32))

        # 填充凸包區域
        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    def _estimate_midline(
        self, face_points: np.ndarray, midline_indices: Optional[Tuple[int, ...]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估計臉部中線（使用 PCA）

        Args:
            face_points: 臉部特徵點
            midline_indices: 中線相關的特徵點索引

        Returns:
            (中線上的點 p0, 法向量 n)
        """
        if midline_indices is None:
            # 使用預設的中線特徵點
            midline_indices = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)

        # 提取中線特徵點
        idx = np.array(midline_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points.shape[0])]

        if idx.size == 0:
            ml_pts = face_points
        else:
            ml_pts = face_points[idx, :]

        # PCA 找主方向
        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0

        # 處理退化情況
        if not np.isfinite(X).all() or np.allclose(X, 0):
            # 使用垂直中線
            xs = face_points[:, 0]
            mid_x = 0.5 * (xs.min() + xs.max())
            p0 = np.array([mid_x, face_points[:, 1].mean()], dtype=np.float64)
            n = np.array([1.0, 0.0], dtype=np.float64)
            return p0, n

        # SVD 分解
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]
        u = u / (np.linalg.norm(u) + 1e-12)

        # 法向量（垂直於主方向）
        n = np.array([-u[1], u[0]], dtype=np.float64)

        # 確保 n 指向右側
        if n[0] < 0:
            n = -n

        return p0, n

    def _align_to_canvas_premul(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        p0: np.ndarray,
        n: np.ndarray,
        side: str,
        out_size: Tuple[int, int],
        margin: float,
    ) -> np.ndarray:
        """
        對齊到畫布並使用預乘 alpha（完整版本）

        Args:
            img_bgr: 輸入影像
            mask_u8: 臉部遮罩
            p0: 中線上的點
            n: 法向量
            side: 'left' 或 'right'
            out_size: 輸出尺寸
            margin: 邊緣留白

        Returns:
            鏡射影像
        """
        H, W = out_size
        h, w = img_bgr.shape[:2]

        # 計算每個像素到中線的有號距離
        X, Y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]

        # 計算反射座標
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]

        # 建立半臉 alpha 遮罩
        if side == "left":
            region = (mask_u8 > 0) & (d < 0)
        else:
            region = (mask_u8 > 0) & (d > 0)

        alpha = np.zeros_like(mask_u8, dtype=np.uint8)
        alpha[region] = 255

        # 羽化邊緣
        if self.config.feather_px > 0:
            kernel_size = self.config.feather_px * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

        alpha_f = alpha.astype(np.float32) / 255.0

        # 反射另一半
        reflected = cv2.remap(img_bgr, Xr, Yr, cv2.INTER_LINEAR)
        reflected_alpha = cv2.remap(alpha_f, Xr, Yr, cv2.INTER_LINEAR)

        # 預乘 alpha 合成
        img_f = img_bgr.astype(np.float32) / 255.0
        result_f = (
            img_f * alpha_f[..., None]
            + (reflected.astype(np.float32) / 255.0) * reflected_alpha[..., None]
        )

        final_alpha = np.clip(alpha_f + reflected_alpha, 0, 1)

        # 除以 alpha 還原顏色
        eps = 1e-6
        result_f = np.where(
            final_alpha[..., None] > eps, result_f / final_alpha[..., None], 0
        )

        result = np.clip(result_f * 255, 0, 255).astype(np.uint8)

        # 找出內容邊界
        alpha_mask = (final_alpha * 255).astype(np.uint8)
        ys, xs = np.where(alpha_mask > 0)

        if len(xs) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # 裁切
        cropped = result[y0 : y1 + 1, x0 : x1 + 1]

        # 計算縮放比例
        face_w = x1 - x0 + 1
        face_h = y1 - y0 + 1

        available_w = W * (1 - 2 * margin)
        available_h = H * (1 - 2 * margin)

        scale = min(available_w / face_w, available_h / face_h, 1.0)

        # 縮放
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 置中到畫布
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        start_x = (W - new_w) // 2
        start_y = (H - new_h) // 2
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized

        return canvas

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        應用 CLAHE 直方圖均衡

        Args:
            image: 輸入影像

        Returns:
            增強後的影像
        """
        # 確保影像格式正確
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 轉換到 Lab 色彩空間
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 建立 CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        )

        # 應用到 L 通道
        l_eq = clahe.apply(l)

        # 合併通道
        lab_eq = cv2.merge([l_eq, a, b])

        # 轉換回 BGR
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        return result

    def _landmarks_to_array(
        self, landmarks, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        將 MediaPipe 特徵點轉換為 numpy 陣列

        Args:
            landmarks: MediaPipe 特徵點
            image_shape: 影像尺寸

        Returns:
            特徵點陣列 (468, 2)
        """
        h, w = image_shape[:2]
        points = []

        for lm in landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            points.append([x, y])

        return np.array(points, dtype=np.float64)

    # ========== 儲存函數 ==========

    def _save_selected(self, faces: List[FaceInfo]):
        """儲存選中的影像"""
        if not self.config.save_intermediate:
            return

        save_dir = self.config.get_workspace_subdir("selected")
        if not save_dir:
            return

        for i, face in enumerate(faces):
            filename = f"selected_{i:03d}_angle_{face.angle:.1f}.png"
            path = save_dir / filename
            cv2.imwrite(str(path), face.image)
            logger.debug(f"儲存選中影像: {path}")

    def _save_aligned(self, image: np.ndarray, face_info: FaceInfo, index: int):
        """儲存對齊後的影像"""
        if not self.config.save_intermediate:
            return

        save_dir = self.config.get_workspace_subdir("aligned")
        if not save_dir:
            return

        if face_info.path:
            filename = f"{face_info.path.stem}_aligned.png"
        else:
            filename = f"aligned_{index:03d}.png"

        path = save_dir / filename
        cv2.imwrite(str(path), image)
        logger.debug(f"儲存對齊影像: {path}")

    def _save_mirrors(
        self, left: np.ndarray, right: np.ndarray, face_info: FaceInfo, index: int
    ):
        """儲存鏡射影像"""
        if not self.config.save_intermediate:
            return

        save_dir = self.config.get_workspace_subdir("mirrors")
        if not save_dir:
            return

        if face_info.path:
            base_name = face_info.path.stem
        else:
            base_name = f"mirror_{index:03d}"

        left_path = save_dir / f"{base_name}_left.png"
        right_path = save_dir / f"{base_name}_right.png"

        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)
        logger.debug(f"儲存鏡射影像: {left_path}, {right_path}")
