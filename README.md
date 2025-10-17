# 人臉不對稱性與認知評估 API

基於深度學習的人臉分析與 6QDS 認知評估系統，使用 Docker 容器化部署。

## 準備工作

### 1️⃣ 準備訓練模型

將相關的模型檔案放入 `model/` 目錄：

```
model/
├── xgb_6qds_model.json              # 6QDS 認知評估模型
├── topofr_average_cdr1.0.json       # 人臉不對稱性分類器
└── topofr_average_cdr1.0_features.json  # 特徵選取配置
```

### 2️⃣ 準備外部依賴

#### Dlib 模型

1. 下載模型檔案到 `external/dlib/`：
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`

   **下載連結**: http://dlib.net/files/

   或使用download_dlib.py下載 

**最終結構**：
```
external/dlib/
├── shape_predictor_68_face_landmarks.dat
└── dlib_face_recognition_resnet_model_v1.dat
```

---

#### TopoFR 模型

1. 克隆 TopoFR 專案到 `external/TopoFR/`：
   ```bash
   cd external
   git clone https://github.com/DanJun6737/TopoFR.git
   ```

2. **替換 backbones/iresnet.py （重要）**
   置換為自定義版本

3. 放置 TopoFR 預訓練權重到 `external/TopoFR/model/`：
   ```
   external/TopoFR/model/
   └── [你的 TopoFR 模型].pt  (例如: TopoFR_r100.pt)
   ```

**最終結構**：
```
external/TopoFR/
├── backbones/           
│   ├── __init__.py
│   └── iresnet.py       # ← 需替換為自定義版本
├── model/
│   └── TopoFR_r100.pt   # ← 預訓練權重
└── [其他 TopoFR 檔案]
```

## Docker 啟動步驟

### 1️⃣ 建置映像檔

```bash
docker-compose build --no-cache
```

**建置時間**: 首次約 10-15 分鐘

---

### 2️⃣ 啟動容器

```bash
# 前景執行（可看到日誌）
docker-compose up

# 或背景執行
docker-compose up -d
```

**啟動時間**: 約 30-60 秒

## API

### 互動式文檔

開啟瀏覽器訪問: **http://localhost:8000/docs**

### API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/health` | GET | 健康檢查 |
| `/analyze` | POST | 上傳壓縮檔與問卷進行分析 |

### 請求範例

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@faces.zip" \
  -F "age=75" \
  -F "gender=1" \
  -F "education_years=12" \
  -F "q1=0" -F "q2=1" -F "q3=0" \
  -F "q4=1" -F "q5=0" -F "q6=1" \
  -F "q7=0" -F "q8=1" -F "q9=0" -F "q10=1"
```

### 輸入要求

- **壓縮檔格式**: `.zip`, `.7z`, `.rar`
- **圖片數量**: 1200 張
- **圖片格式**: JPG, PNG, BMP, TIFF
- **圖片內容**: 正面人臉照片
- **檔案大小**: ≤ 500MB

---