"""
核心配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class PreprocessConfig:
    """共用預處理配置"""

    # ========== 相片選擇參數 ==========
    n_select: int = 10  # 選擇多少張最正的臉部相片
    detection_confidence: float = 0.5  # MediaPipe 偵測信心度閾值

    # ========== 角度校正參數 ==========
    align_face: bool = True  # 是否校正角度

    # ========== 鏡射參數 ==========
    mirror_size: Tuple[int, int] = (512, 512)  # 輸出鏡射影像大小
    feather_px: int = 2  # 邊緣羽化像素
    margin: float = 0.08  # 畫布邊緣留白比例

    # ========== CLAHE 參數 ==========
    apply_clahe: bool = True  # 是否應用 CLAHE
    clahe_clip_limit: float = 2.0  # CLAHE 限制參數
    clahe_tile_size: int = 8  # CLAHE 區塊大小

    # ========== 儲存控制 ==========
    save_intermediate: bool = False  # 是否儲存中間結果
    workspace_dir: Optional[Path] = None  # 工作區路徑
    subject_id: Optional[str] = None  # 受試者 ID

    # ========== 處理流程控制 ==========
    steps: List[str] = field(
        default_factory=lambda: [
            "select",  # 選擇最正面的 n 張
            "align",  # 角度校正
            "mirror",  # 生成鏡射
            "clahe",  # CLAHE 增強
        ]
    )

    def __post_init__(self):
        """初始化後處理"""
        # 如果要儲存但沒指定路徑，使用預設路徑
        if self.save_intermediate and not self.workspace_dir:
            self.workspace_dir = Path("workspace")

        # 確保路徑是 Path 物件
        if self.workspace_dir:
            self.workspace_dir = Path(self.workspace_dir)

    def get_workspace_subdir(self, subdir_name: str) -> Optional[Path]:
        """
        取得工作區子目錄路徑
        
        Args:
            subdir_name: 子目錄名稱 (selected, aligned, mirrors, debug)
            
        Returns:
            完整路徑，格式為 workspace_dir/subdir_name/subject_id/
        """
        if not self.workspace_dir:
            return None
        
        if self.subject_id:
            path = self.workspace_dir / subdir_name / self.subject_id
        else:
            path = self.workspace_dir / subdir_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class APIConfig(PreprocessConfig):
    """API 配置"""

    save_intermediate: bool = False  # API 預設不儲存
    workspace_dir: Optional[Path] = field(
        default_factory=lambda: Path("workspace/temp")
    )
    cleanup_on_complete: bool = True  # 完成後清理暫存檔


@dataclass
class AnalyzeConfig(PreprocessConfig):
    """Analyze 配置"""

    save_intermediate: bool = True  # Analyze 預設儲存
    workspace_dir: Optional[Path] = field(
        default_factory=lambda: Path("workspace/analysis")
    )
    experiment_name: Optional[str] = None  # 實驗名稱（用於組織輸出）

    def __post_init__(self):
        super().__post_init__()
        # 如果有實驗名稱，加到路徑中
        if self.experiment_name and self.workspace_dir:
            self.workspace_dir = self.workspace_dir / self.experiment_name
