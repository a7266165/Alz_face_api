"""
核心臉部預處理

移植自 Alz_face_analyze/src/preprocess（detect → select → [mask if no_background] → align → mirror），
維持與訓練端逐字一致的幾何數學。

偵測層用外部 landmark 服務（mediapipe `mp.solutions`，跑在 numpy<2 子服務；見
`landmark_service/`）——solutions 需 numpy<2、與模型 pickle(numpy2) 互斥，故隔離成子服務。
選圖/對齊/裁切的 landmark 皆打這個服務（與訓練端 run_preprocess / crop_faces.py 同一偵測器）；
幾何運算（選圖/對齊/鏡射/裁切）在本模組(numpy2) 完成。

對外只暴露 FacePreprocessor.process(images) -> List[ProcessedFace]，每筆含：
  - aligned      : 對齊後「全幀」影像（背景版）→ LR#1 的 ArcFace embedding
  - age_crop     : 在 aligned 上重偵測 landmark、取外接框 + margin 裁出的臉特寫
                   → MiVOLO 年齡預測（對齊 analyze scripts/age/crop_faces.py）
  - mirror_left  : resize_and_center(aligned) 的 512×512 畫布（face_left）
  - mirror_right : mirror_left 的水平翻轉（flip 法；face_right）→ LR#2 差 D = E(left) − E(right)
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """單張臉部偵測資訊。"""

    image: np.ndarray              # 原始 BGR 影像
    vertex_angle_sum: float        # 中軸線頂點夾角總和（度），越小越正面
    landmarks: np.ndarray          # 特徵點座標 (N, 2)


@dataclass
class ProcessedFace:
    """單張臉部預處理結果。"""

    aligned: np.ndarray            # 對齊後全幀（背景版）— LR#1 embedding
    age_crop: np.ndarray           # aligned 上 landmark 外接框 + margin 的臉特寫 — MiVOLO 年齡
    mirror_left: np.ndarray        # 512 畫布原圖（face_left）— LR#2 差的左項
    mirror_right: np.ndarray       # 512 畫布鏡射（face_right）— LR#2 差的右項


class FacePreprocessor:
    """臉部預處理器（landmark 偵測走 landmark_service）。"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._service_url = self.config.landmark_service_url
        self._service_timeout = self.config.landmark_service_timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    # ========== 主流程 ==========

    def process(self, images: List[np.ndarray]) -> List[ProcessedFace]:
        """detect → select(n) → [mask if no_background] → align → flip-mirror。"""
        if not images:
            logger.warning("沒有輸入影像")
            return []

        logger.info(f"開始預處理 {len(images)} 張影像")

        face_infos = self._detect_faces(images)
        logger.info(f"偵測到 {len(face_infos)} 張臉")
        if not face_infos:
            return []

        selected = self._select_most_frontal(face_infos)
        logger.info(f"選出最正面的 {len(selected)} 張")

        processed: List[ProcessedFace] = []
        for info in selected:
            try:
                processed.append(self._process_single(info))
            except Exception as e:  # 單張失敗不影響整批
                logger.error(f"處理單張臉部失敗: {e}")
        logger.info(f"預處理完成，共 {len(processed)} 張")
        return processed

    def _process_single(self, info: FaceInfo) -> ProcessedFace:
        image = info.image
        if self.config.apply_mask:                       # no_background → 凸包去背
            image = self._apply_mask(image, info.landmarks)

        tilt = self._calculate_midline_tilt(info.landmarks)
        aligned = self._rotate_to_vertical(image, tilt)  # 全幀（背景版）

        age_crop = self._crop_face_for_age(aligned)      # 臉特寫 → MiVOLO
        mirror_left, mirror_right = self._generate_mirrors(aligned, info.landmarks)
        return ProcessedFace(
            aligned=aligned,
            age_crop=age_crop,
            mirror_left=mirror_left,
            mirror_right=mirror_right,
        )

    # ========== 年齡分支臉部裁切 ==========

    def _crop_face_for_age(self, aligned: np.ndarray) -> np.ndarray:
        """複製 analyze scripts/age/crop_faces.py：在對齊圖上重偵測 landmark，
        取所有點外接框 + margin 外擴後裁切，供 MiVOLO 年齡預測。

        偵測不到臉 → 退回整張對齊圖（與 crop_faces.py 的 fallback 一致：臉已置中）。
        """
        pts = self._detect_landmarks(aligned)
        if pts is None:
            logger.debug("年齡裁切：對齊圖未偵測到臉，退回整張對齊圖")
            return aligned
        return self._crop_to_landmarks(aligned, pts, self.config.age_crop_margin)

    @staticmethod
    def _crop_to_landmarks(image: np.ndarray, landmarks: np.ndarray,
                           margin: float) -> np.ndarray:
        """landmarks(N,2 像素座標)外接框 + margin 外擴後裁切；margin 以框長邊為基準。

        逐字對齊 analyze crop_faces.py:crop_to_landmarks（含 int 轉換與邊界裁剪）。
        """
        h, w = image.shape[:2]
        x1, y1 = landmarks[:, 0].min(), landmarks[:, 1].min()
        x2, y2 = landmarks[:, 0].max(), landmarks[:, 1].max()
        m = margin * max(x2 - x1, y2 - y1)
        X1, Y1 = int(max(0, x1 - m)), int(max(0, y1 - m))
        X2, Y2 = int(min(w, x2 + m)), int(min(h, y2 + m))
        return image[Y1:Y2, X1:X2]

    # ========== detect / select ==========

    def _detect_faces(self, images: List[np.ndarray]) -> List[FaceInfo]:
        midline = self.config.mirror.midline_points
        out: List[FaceInfo] = []
        for i, image in enumerate(images):
            points = self._detect_landmarks(image)
            if points is None:
                logger.debug(f"第 {i} 張未偵測到臉")
                continue
            out.append(FaceInfo(
                image=image,
                vertex_angle_sum=self._vertex_angle_sum(points, midline),
                landmarks=points,
            ))
        return out

    def _detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """單張 BGR → (N, 2) 像素座標特徵點；偵測不到回 None。

        經外部 landmark 服務（mp.solutions，numpy<2）偵測：傳原始 BGR bytes（不重編碼）。
        服務為硬依賴，不可達時丟 RuntimeError。
        """
        img = np.ascontiguousarray(image, dtype=np.uint8)
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        req = urllib.request.Request(
            self._service_url,
            data=img.tobytes(),
            method="POST",
            headers={"X-Height": str(h), "X-Width": str(w), "X-Channels": str(c)},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._service_timeout) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(
                f"landmark 服務無法連線（{self._service_url}）：{e}；"
                f"請確認子服務已啟動（landmark_service/server.py，numpy<2 環境）。"
            ) from e
        if not result.get("detected"):
            return None
        return np.asarray(result["landmarks"], dtype=np.float64)

    def _select_most_frontal(self, face_infos: List[FaceInfo]) -> List[FaceInfo]:
        """按頂點夾角總和（恆非負，越小越正面）升冪取前 n 張。"""
        n = self.config.n_select
        return sorted(face_infos, key=lambda x: x.vertex_angle_sum)[:min(n, len(face_infos))]

    @staticmethod
    def _vertex_angle_sum(points: np.ndarray, midline_points: Tuple[int, ...]) -> float:
        """中軸線頂點夾角總和（度）：折線各頂點相鄰線段夾角之和。"""
        dots = [points[i] for i in midline_points]
        v1, v2, v3 = dots[1] - dots[0], dots[2] - dots[1], dots[3] - dots[2]

        def ang(a, b):
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return np.arccos(np.clip(np.dot(a, b) / (norm + 1e-8), -1.0, 1.0))

        return float(np.degrees(ang(v1, v2)) + np.degrees(ang(v2, v3)))

    # ========== mask / align ==========

    @staticmethod
    def _apply_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """去背：臉部凸包以外塗黑。"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if landmarks.shape[0] > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(landmarks.astype(np.int32)), 255)
        return cv2.bitwise_and(image, image, mask=mask)

    def _calculate_midline_tilt(self, landmarks: np.ndarray) -> float:
        """中軸線相對垂直線的傾斜角（度）；正值=向右傾。"""
        pts = self.config.mirror.midline_points
        angles = []
        for i in range(len(pts) - 1):
            x1, y1 = landmarks[pts[i]]
            x2, y2 = landmarks[pts[i + 1]]
            dx, dy = x2 - x1, y2 - y1
            if abs(dy) < 1e-8:
                angles.append(90.0 if dx > 0 else -90.0)
            else:
                angles.append(np.degrees(np.arctan(dx / dy)))
        return float(np.mean(angles)) if angles else 0.0

    @staticmethod
    def _rotate_to_vertical(image: np.ndarray, tilt: float) -> np.ndarray:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    # ========== mirror ==========

    def _generate_mirrors(self, image: np.ndarray,
                          landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config.mirror
        if cfg.mirror_method == "flip":
            original = self._resize_and_center(image, cfg.mirror_size, cfg.margin)
            return original, cv2.flip(original, 1)
        # midline：保留路徑
        p0, n = self._estimate_midline(landmarks, cfg.midline_points)
        left = self._align_to_canvas_premul(image, p0, n, "left", cfg.mirror_size,
                                            cfg.feather_px, cfg.margin)
        right = self._align_to_canvas_premul(image, p0, n, "right", cfg.mirror_size,
                                             cfg.feather_px, cfg.margin)
        return left, right

    @staticmethod
    def _resize_and_center(image: np.ndarray, out_size: Tuple[int, int],
                           margin: float) -> np.ndarray:
        """縮放並置中到畫布（裁掉旋轉後的純黑邊，保留留白）。"""
        H, W = out_size
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 0)
        if len(xs) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cropped = image[y0:y1 + 1, x0:x1 + 1]
        face_h, face_w = cropped.shape[:2]

        scale = min(W * (1 - 2 * margin) / face_w, H * (1 - 2 * margin) / face_h, 1.0)
        new_w, new_h = int(face_w * scale), int(face_h * scale)
        if new_w <= 0 or new_h <= 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        sx, sy = (W - new_w) // 2, (H - new_h) // 2
        canvas[sy:sy + new_h, sx:sx + new_w] = resized
        return canvas

    @staticmethod
    def _estimate_midline(face_points: np.ndarray,
                          midline_indices: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.array(midline_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points.shape[0])]
        ml_pts = face_points if idx.size == 0 else face_points[idx, :]

        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0
        if not np.isfinite(X).all() or np.allclose(X, 0):
            xs = face_points[:, 0]
            p0 = np.array([0.5 * (xs.min() + xs.max()), face_points[:, 1].mean()], dtype=np.float64)
            return p0, np.array([1.0, 0.0], dtype=np.float64)

        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
        n = np.array([-u[1], u[0]], dtype=np.float64)
        if n[0] < 0:
            n = -n
        return p0, n

    def _align_to_canvas_premul(self, img_bgr: np.ndarray, p0: np.ndarray, n: np.ndarray,
                                side: str, out_size: Tuple[int, int],
                                feather_px: int, margin: float) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]

        half_mask = ((d < 0) if side == "left" else (d > 0)).astype(np.uint8) * 255
        if feather_px > 0:
            k = feather_px * 2 + 1
            half_mask = cv2.GaussianBlur(half_mask, (k, k), 0)
        alpha_f = half_mask.astype(np.float32) / 255.0

        reflected = cv2.remap(img_bgr, Xr, Yr, cv2.INTER_LINEAR)
        reflected_alpha = cv2.remap(alpha_f, Xr, Yr, cv2.INTER_LINEAR)

        img_f = img_bgr.astype(np.float32) / 255.0
        result_f = (img_f * alpha_f[..., None]
                    + (reflected.astype(np.float32) / 255.0) * reflected_alpha[..., None])
        final_alpha = np.clip(alpha_f + reflected_alpha, 0, 1)
        result_f = np.where(final_alpha[..., None] > 1e-6, result_f / final_alpha[..., None], 0)
        result = np.clip(result_f * 255, 0, 255).astype(np.uint8)
        return self._resize_and_center(result, out_size, margin)
