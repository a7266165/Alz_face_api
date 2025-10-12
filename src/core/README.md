# Core 模組

本模組提供 API 和 Analyze 專案共用的核心功能。

## 模組結構

- `config.py`: 配置管理
- `preprocess.py`: 影像預處理
- `feature_extract.py`: 特徵提取（待實作）

## 使用方式

### API 專案
```python
from src.core import APIConfig, FacePreprocessor

config = APIConfig(n_select=10, save_intermediate=False)
with FacePreprocessor(config) as preprocessor:
    results = preprocessor.process(images)