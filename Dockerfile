# ===================================
# Dockerfile - 統一版本（支援 GPU/CPU）
# ===================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 避免互動式提示
ENV DEBIAN_FRONTEND=noninteractive
# 預設時區（避免 tzdata 詢問）
ENV TZ=Asia/Taipei

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    tzdata \
    build-essential \
    cmake \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    p7zip-full \
    unar \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 設定 Python 3.11 為預設
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升級 pip
RUN python -m pip install --upgrade pip

WORKDIR /app

# 複製依賴檔案
COPY pyproject.toml poetry.lock* ./

# 安裝 Poetry
RUN pip install --no-cache-dir poetry==2.1.3 && \
    poetry config virtualenvs.create false

# 安裝專案依賴（直接使用 poetry.lock）
RUN poetry install --no-root --only main --no-interaction --no-ansi

# 複製應用程式碼
COPY . .

# 建立必要目錄
RUN mkdir -p model workspace/temp

# 環境變數
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "app.py"]