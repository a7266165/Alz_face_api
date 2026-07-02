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

## 法二備註

`requirements.txt` 是為「Linux/Windows x86_64 + Python 3.11 + 有 build 工具 + setuptools<81」
寫的，這些前置 **Dockerfile 已封裝好**；在**別台機器裸機** `pip install -r requirements.txt`
少了前置，常在以下三處其中之一失敗。**換機器建議直接用法一（Docker），只需 Docker + 模型。**

1. **`mivolo`（git 依賴）需要舊 setuptools 與 git。**
   其 `setup.py` 會 `import pkg_resources`，但 setuptools ≥ 81 已移除 →
   `ModuleNotFoundError: No module named 'pkg_resources'`。且 git+ 依賴需系統有 `git`。
   修法：先 `pip install "setuptools<81" wheel`，並以 `PIP_CONSTRAINT` 鎖住再裝（Dockerfile 即此法）。

2. **`insightface==0.7.3` 多數平台無預編譯 wheel，需現場編譯。**
   需 C++ 編譯器 + cmake（Windows 缺會報 `Microsoft Visual C++ 14.0 is required`；
   Linux 缺 gcc/cmake 或 `Python.h`）。修法：Windows 裝 VS Build Tools、Linux 裝 `build-essential cmake`。

3. **`torch==2.7.1+cpu` / `torchvision==0.22.1+cpu` 綁平台與 Python 版本。**
   `+cpu` wheel 只在 PyTorch CPU index、且只有 Linux x86_64 / Windows 有 →
   macOS / Linux ARM 會 `No matching distribution found`。整份 pin 也假設 Python 3.11 x86_64。

