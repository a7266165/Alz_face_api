"""
輸入處理：單一上傳檔 → BGR 影像列表

/analyze 接受「單一檔案」：
  - 影片：單一 .mp4 → cv2 抽所有幀
  - 壓縮檔：.zip / .7z / .rar（內含 n 張 .jpg/.jpeg/.png/.bmp/.tiff）→ 解壓後讀目錄裡所有圖

兩條路徑都統一輸出 List[np.ndarray]（BGR），交給預處理選最正面的 10 張。
壓縮檔只是換一個入口；下游 pipeline（preprocess → age → arcface → LR → TabPFN）完全不變。
"""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 可選壓縮格式依賴（.7z 需 py7zr；.rar 需 rarfile + 系統 unar/unrar）
try:
    import py7zr

    HAS_7Z = True
except ImportError:
    HAS_7Z = False

try:
    import rarfile

    HAS_RAR = True
except ImportError:
    HAS_RAR = False

VIDEO_EXTS = {".mp4"}
ARCHIVE_EXTS = {".zip", ".7z", ".rar"}
# 壓縮檔內可讀的圖片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class FileHandler:
    """把單一上傳檔（.mp4 或壓縮檔）解碼成 BGR 影像列表。"""

    # ---- 能力查詢 -------------------------------------------------------

    @staticmethod
    def supported_archive_formats() -> List[str]:
        """目前環境實際可解的壓縮格式（依已安裝套件而定）。"""
        formats = [".zip"]  # zipfile 內建，恆可用
        if HAS_7Z:
            formats.append(".7z")
        if HAS_RAR:
            formats.append(".rar")
        return formats

    # ---- 影片解碼 -------------------------------------------------------

    @staticmethod
    def extract_video_frames(data: bytes, suffix: str = ".mp4") -> List[np.ndarray]:
        """影片 bytes → 抽「所有」幀（BGR）。"""
        frames: List[np.ndarray] = []
        # cv2.VideoCapture 需要實體檔；寫到暫存檔後讀取。
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                raise ValueError("無法開啟影片（編碼不支援或檔案損毀）")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
        finally:
            tmp_path.unlink(missing_ok=True)
        logger.info(f"影片抽出 {len(frames)} 幀")
        if not frames:
            raise ValueError("影片未抽出任何幀")
        return frames

    # ---- 壓縮檔 ---------------------------------------------------------

    @classmethod
    def _extract_archive(cls, archive_path: Path, extract_to: Path) -> None:
        """依副檔名解壓到 extract_to。"""
        ext = archive_path.suffix.lower()
        if ext == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
        elif ext == ".7z":
            if not HAS_7Z:
                raise ValueError("伺服器未安裝 py7zr，無法解 .7z")
            with py7zr.SevenZipFile(archive_path, mode="r") as szf:
                szf.extractall(extract_to)
        elif ext == ".rar":
            if not HAS_RAR:
                raise ValueError("伺服器未安裝 rarfile / unar，無法解 .rar")
            with rarfile.RarFile(archive_path, "r") as rf:
                rf.extractall(extract_to)
        else:
            raise ValueError(f"不支援的壓縮格式: {ext}")

    @classmethod
    def _load_images_from_dir(cls, root: Path) -> List[np.ndarray]:
        """遞迴讀取目錄裡所有圖片（依路徑排序，固定順序）。"""
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        images: List[np.ndarray] = []
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"無法讀取圖片: {p.name}")
                continue
            images.append(img)
        return images

    @classmethod
    def load_archive(cls, data: bytes, suffix: str) -> List[np.ndarray]:
        """壓縮檔 bytes → 解壓 → 讀目錄裡所有圖 → BGR 影像列表。

        解壓與讀檔都在 TemporaryDirectory 內完成，結束自動清除。
        """
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            archive_path = tdir / f"upload{suffix}"
            archive_path.write_bytes(data)
            extract_dir = tdir / "extracted"
            extract_dir.mkdir()
            try:
                cls._extract_archive(archive_path, extract_dir)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"解壓縮失敗: {e}")
            images = cls._load_images_from_dir(extract_dir)
        if not images:
            raise ValueError("壓縮檔中未找到可讀取的圖片")
        logger.info(f"壓縮檔載入 {len(images)} 張圖片")
        return images

    # ---- 統一入口 -------------------------------------------------------

    @classmethod
    def load_single(cls, filename: str, data: bytes) -> List[np.ndarray]:
        """把單一上傳檔統一成 BGR 影像列表。

        Args:
            filename: 原始檔名（用副檔名判斷型別）
            data: 檔案位元組內容

        Returns:
            BGR 影像列表。

        Raises:
            ValueError: 格式不支援、未安裝對應解壓套件、或解碼/解壓失敗。
        """
        ext = Path(filename or "upload").suffix.lower()

        if ext in VIDEO_EXTS:
            return cls.extract_video_frames(data, suffix=ext)

        if ext in ARCHIVE_EXTS:
            if ext not in cls.supported_archive_formats():
                raise ValueError(f"伺服器未安裝對應解壓支援，無法處理 {ext}")
            return cls.load_archive(data, suffix=ext)

        raise ValueError(
            f"不支援的格式: {ext}；支援 .mp4 或壓縮檔 "
            f"{', '.join(cls.supported_archive_formats())}"
        )
