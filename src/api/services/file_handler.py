"""
src/api/services/file_handler.py
檔案處理服務：解壓縮、圖片載入
"""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 檢查可選依賴
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


class FileHandler:
    """檔案處理器"""
    
    SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def __init__(self):
        """初始化檔案處理器"""
        self._log_capabilities()
    
    def _log_capabilities(self):
        """記錄支援的格式"""
        formats = ['.zip']
        if HAS_7Z:
            formats.append('.7z')
        if HAS_RAR:
            formats.append('.rar')
        logger.info(f"支援的壓縮格式: {', '.join(formats)}")
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """取得支援的壓縮格式"""
        formats = ['.zip']
        if HAS_7Z:
            formats.append('.7z')
        if HAS_RAR:
            formats.append('.rar')
        return formats
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> Path:
        """
        解壓縮檔案
        
        Args:
            archive_path: 壓縮檔路徑
            extract_to: 解壓縮目標目錄
            
        Returns:
            包含圖片的目錄路徑
            
        Raises:
            ValueError: 不支援的格式或解壓縮失敗
        """
        ext = archive_path.suffix.lower()
        
        logger.info(f"解壓縮 {archive_path.name} 到 {extract_to}")
        
        try:
            if ext == '.zip':
                self._extract_zip(archive_path, extract_to)
            elif ext == '.7z':
                self._extract_7z(archive_path, extract_to)
            elif ext == '.rar':
                self._extract_rar(archive_path, extract_to)
            else:
                raise ValueError(f"不支援的壓縮格式: {ext}")
        except Exception as e:
            raise ValueError(f"解壓縮失敗: {e}")
        
        # 找到包含圖片的目錄
        image_dir = self._find_image_dir(extract_to)
        if not image_dir:
            raise ValueError("壓縮檔中未找到圖片檔案")
        
        logger.info(f"找到圖片目錄: {image_dir}")
        return image_dir
    
    def _extract_zip(self, archive_path: Path, extract_to: Path):
        """解壓縮 ZIP"""
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    
    def _extract_7z(self, archive_path: Path, extract_to: Path):
        """解壓縮 7Z"""
        if not HAS_7Z:
            raise ValueError("需要安裝 py7zr: pip install py7zr")
        with py7zr.SevenZipFile(archive_path, mode='r') as szf:
            szf.extractall(extract_to)
    
    def _extract_rar(self, archive_path: Path, extract_to: Path):
        """解壓縮 RAR"""
        if not HAS_RAR:
            raise ValueError("需要安裝 rarfile: pip install rarfile")
        with rarfile.RarFile(archive_path, 'r') as rf:
            rf.extractall(extract_to)
    
    def _find_image_dir(self, root_dir: Path) -> Path:
        """找到包含圖片的目錄"""
        for image_path in root_dir.rglob('*'):
            if image_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTS:
                return image_path.parent
        return None
    
    def load_images(self, image_dir: Path) -> List[np.ndarray]:
        """
        載入目錄中的所有圖片
        
        Args:
            image_dir: 圖片目錄
            
        Returns:
            圖片列表（BGR 格式）
        """
        images = []
        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])
        
        logger.info(f"載入 {len(image_files)} 張圖片")
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                else:
                    logger.warning(f"無法讀取圖片: {img_path.name}")
            except Exception as e:
                logger.warning(f"讀取圖片失敗 {img_path.name}: {e}")
        
        if not images:
            raise ValueError("未能載入任何有效圖片")
        
        logger.info(f"成功載入 {len(images)} 張圖片")
        return images
    
    @staticmethod
    def create_temp_dir() -> Tuple[tempfile.TemporaryDirectory, Path]:
        """
        建立臨時目錄
        
        Returns:
            (臨時目錄物件, 路徑)
        """
        temp_dir = tempfile.TemporaryDirectory()
        return temp_dir, Path(temp_dir.name)