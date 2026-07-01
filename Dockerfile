# ===================================
# Dockerfile — CPU-only + Python 3.11
# 兩個環境：主 API（numpy2，模型棧）+ landmark 子服務（numpy<2，mp.solutions）。
# 純 CPU 部署：torch/onnxruntime 走 CPU wheel，ArcFace 自動用 CPUExecutionProvider、
# MiVOLO 走 CPU float32、TabPFN 載入時映射至 CPU（皆已在程式內處理，無需改 code）。
# ===================================

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# ffmpeg 供 cv2.VideoCapture 解 .mp4；libgl/glib 等供 opencv/mediapipe；
# unar 供 rarfile 解 .rar；build-essential/cmake/git 供部分套件與 mivolo(git) 安裝；
# curl 供 healthcheck。
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    ffmpeg \
    unar \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app

# ---- 主環境（系統 python，numpy2，CPU torch）：模型棧 ----
# mivolo(git) 的 setup.py 需要 pkg_resources（新 setuptools 已移除）；
# 用 PIP_CONSTRAINT 強制 build 隔離環境用 setuptools<81。
COPY requirements.txt ./
RUN printf 'setuptools<81\n' > /tmp/build-constraints.txt && \
    pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    PIP_CONSTRAINT=/tmp/build-constraints.txt pip install --no-cache-dir -r requirements.txt

# ---- 第二環境（/opt/lmkenv，numpy<2）：landmark 子服務，與主環境完全隔離 ----
COPY landmark_service/requirements.txt /tmp/lmk_req.txt
RUN python -m venv /opt/lmkenv && \
    /opt/lmkenv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/lmkenv/bin/pip install --no-cache-dir -r /tmp/lmk_req.txt

COPY . .

RUN mkdir -p model && chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
# 主 API 用這個 URL 打 landmark 子服務（同容器內 localhost）。
ENV LANDMARK_SERVICE_URL=http://127.0.0.1:8771/landmarks

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# entrypoint：先起 landmark 子服務（健康後）再起主 API。
CMD ["/app/entrypoint.sh"]
