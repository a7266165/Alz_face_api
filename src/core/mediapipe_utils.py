"""
MediaPipe Face Mesh landmark 常數。

與 Alz_face_analyze/src/common/mediapipe_utils.py 同步：本 API 的預處理只用到
中軸線 4 點（正面判定的頂點夾角 + 對齊傾角），故此處僅保留 MIDLINE_POINTS。
embedding 模態的左右不對稱以「原圖 vs 水平翻轉」的 ArcFace embedding 差求得，
不依賴雙側 landmark，因此不需要 LEFT/RIGHT_FACE_INDICES 等常數。
"""

from typing import Tuple

# 4 點中軸線 — 用於臉部正面判定（頂點夾角）與對齊傾角計算。
MIDLINE_POINTS: Tuple[int, ...] = (10, 168, 4, 2)
