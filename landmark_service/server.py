"""Landmark 偵測子服務(舊 mediapipe `mp.solutions`,跑在 numpy<2 環境)。

mp.solutions 偵測器需 numpy<2，與部署模型 pickle(numpy2) 互斥，故隔離成這個 localhost
子服務：主 API(numpy2) 透過 HTTP 拿座標，預處理的選圖/對齊/裁切共用同一偵測器
（與訓練端 crop_faces.py / run_preprocess 同一套）。

協定（內部、localhost）：
  POST /landmarks   body = 原始 BGR uint8 bytes（不重編碼，省大量影格的開銷）
                    headers: X-Height / X-Width / X-Channels(=3)
                    → {"detected": bool, "landmarks": [[x,y],...]}
  GET  /health      → {"status": "ok", ...}

啟動：python landmark_service/server.py   （在 numpy<2 + mediapipe 0.10.21 環境）
埠由環境變數 LANDMARK_SERVICE_PORT 決定（預設 8771）。
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import mediapipe as mp

PORT = int(os.environ.get("LANDMARK_SERVICE_PORT", "8771"))

# 與 Alz_face_analyze/scripts/age/crop_faces.py + run_preprocess 逐字相同的設定
_FM = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,            # 478 點（含 iris）
    min_detection_confidence=0.5,
)
_LOCK = threading.Lock()              # mp.solutions FaceMesh 非執行緒安全 → 推論序列化


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):     # 靜音預設 access log
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "mediapipe": mp.__version__,
                        "numpy": np.__version__})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            h = int(self.headers["X-Height"])
            w = int(self.headers["X-Width"])
            c = int(self.headers.get("X-Channels", 3))
            raw = self.rfile.read(n)
            img = np.frombuffer(raw, np.uint8).reshape(h, w, c)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with _LOCK:
                res = _FM.process(rgb)
            if not res.multi_face_landmarks:
                self._send({"detected": False})
                return
            lm = res.multi_face_landmarks[0].landmark
            self._send({"detected": True,
                        "landmarks": [[p.x * w, p.y * h] for p in lm]})
        except Exception as e:        # 壞請求/解碼失敗 → 回 500，主端會丟清楚的錯
            self._send({"detected": False, "error": str(e)}, 500)


if __name__ == "__main__":
    print(f"[landmark_service] mp.solutions {mp.__version__} / numpy {np.__version__} "
          f"on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
