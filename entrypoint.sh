#!/usr/bin/env bash
# 容器入口：先起 landmark 子服務（numpy<2 venv），健康後再起主 API（numpy2）。
# landmark 服務為硬依賴——起不來就 fail fast，不讓主 API 在每次請求才壞。
set -euo pipefail

PORT="${LANDMARK_SERVICE_PORT:-8771}"

echo "[entrypoint] 啟動 landmark 子服務 (mp.solutions, numpy<2, port=$PORT)..."
LANDMARK_SERVICE_PORT="$PORT" /opt/lmkenv/bin/python /app/landmark_service/server.py &
LMK_PID=$!

echo "[entrypoint] 等待 landmark 子服務 health (pid=$LMK_PID)..."
ready=0
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$LMK_PID" 2>/dev/null; then
        echo "[entrypoint] landmark 子服務啟動失敗（process 已退出）" >&2
        exit 1
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "[entrypoint] landmark 子服務 60s 內未健康，中止" >&2
    exit 1
fi
echo "[entrypoint] landmark 子服務 ready"

echo "[entrypoint] 啟動主 API (numpy2)..."
exec python app.py
