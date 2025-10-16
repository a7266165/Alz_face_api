# ===================================
# Dockerfile - CPU 版本
# 修正 Debian Trixie 套件問題
# ===================================

FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    # OpenCV 依賴（新版 Debian 套件名稱）
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # dlib 依賴
    libgomp1 \
    # 壓縮檔支援
    p7zip-full \
    unar \
    # 工具
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製依賴檔案
COPY pyproject.toml poetry.lock* ./

# 安裝 Poetry（輕量級方式）
RUN pip install --no-cache-dir poetry==2.1.3 && \
    poetry config virtualenvs.create false

# 安裝專案依賴
RUN poetry install --no-root --only main --no-interaction --no-ansi

# 複製應用程式碼
COPY . .

# 建立必要目錄
RUN mkdir -p model workspace/temp

# 環境變數
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 啟動應用程式
CMD ["python", "app.py"]