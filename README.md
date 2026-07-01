# 人臉失智評估 API

上傳 `.mp4` 或n張人臉照片的壓縮檔（`.zip`/`.7z`/`.rar`，）＋ 真實年齡，
回傳`predicted_age`、`embedding_LR_score`、`ad_prob`。

## 前置：放入部署模型

模型需另置於根目錄底下的資料夾 model/

```
model/
├── embedding/fold_0..9.joblib   # LR#1（10 折集成）
├── asymmetry/fold_0..9.joblib   # LR#2（10 折集成）
└── tabpfn_core3.pkl             # TabPFN(core3)
```

## 啟動

### 法一、Docker

```bash
docker compose build      # 首次較久（映像約 5.5GB）
docker compose up -d      # → http://localhost:8000/docs
```

### 法二、準備兩個 Python 環境並依序啟動服務

```bash
# 1) landmark 子服務（環境依賴：Alz_face_api\landmark_service\requirements.txt）
python landmark_service/server.py               # → 127.0.0.1:8771

# 2) 主 API（環境依賴：Alz_face_api\requirements.txt）
python app.py                                   # → http://localhost:8000/docs
```

主 API 以環境變數 `LANDMARK_SERVICE_URL`（預設 `http://127.0.0.1:8771/landmarks`）
連子服務。

## 呼叫

```bash
# 壓縮檔或影片擇一
curl -X POST http://localhost:8000/analyze \
  -F "file=@subject.7z" -F "real_age=75"
```

