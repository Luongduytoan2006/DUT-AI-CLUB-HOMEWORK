# -*- coding: utf-8 -*-
"""Web YOLOv1 tren Jetson: stream webcam (2 luong cho muot) + upload anh.
Chay: python3 cam_server.py  ->  http://localhost:3636
"""
import os
import time
import base64
import threading
import cv2
import numpy as np

from detector import Detector, draw, CLASS_NAMES

try:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
except ImportError:                       # Python 3.6
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(HERE, "yolov1_garbage.onnx")
SAVE_DIR = os.path.join(HERE, "captures")
MAX_IMAGES = 10000
PORT = 3636
CAM_INDEX = 0
STREAM_FPS = 15                           # fps hien thi (luong doc cam)


class CameraWorker(object):
    """2 luong: (A) doc cam nhanh de stream muot; (B) infer cham cap nhat box."""
    def __init__(self, detector):
        self.det = detector
        self.cap = None                   # chi mo khi co nguoi xem stream
        self.lock = threading.Lock()
        self.frame = None                 # frame BGR moi nhat
        self.boxes = []                   # ket qua infer gan nhat
        self.last_save = 0.0
        self.viewers = 0                  # so client dang xem stream
        self.last_seen = 0.0              # lan cuoi co nguoi xem

    def start(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def add_viewer(self):
        with self.lock:
            self.viewers += 1
            self.last_seen = time.time()

    def remove_viewer(self):
        with self.lock:
            self.viewers = max(0, self.viewers - 1)

    def _reader(self):
        """Luong A: chi mo cam khi co nguoi xem; het nguoi -> dong cam."""
        while True:
            with self.lock:
                want = self.viewers > 0
            if want and self.cap is None:
                self._open_cam()
            elif not want and self.cap is not None:
                self._close_cam()
            if self.cap is None:
                time.sleep(0.2)
                continue
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            with self.lock:
                self.frame = f

    def _open_cam(self):
        cap = cv2.VideoCapture(CAM_INDEX)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap = cap
        print("Camera MO (co nguoi xem)")

    def _close_cam(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = None
        with self.lock:
            self.frame = None
        print("Camera DONG (het nguoi xem)")

    def _infer_loop(self):
        """Luong B: chay model tren frame moi nhat, cap nhat box (cham cung khong ket stream)."""
        while True:
            with self.lock:
                f = None if self.frame is None else self.frame.copy()
            if f is None:
                time.sleep(0.05)
                continue
            boxes = self.det.infer(f)
            with self.lock:
                self.boxes = boxes
            # luu anh co vat the, toi da ~1 anh/giay
            now = time.time()
            if boxes and (now - self.last_save) > 1.0:
                self._save(f, boxes)
                self.last_save = now

    def _save(self, frame, boxes):
        img = frame.copy()
        draw(img, boxes)
        fn = os.path.join(SAVE_DIR, "det_%d.jpg" % int(time.time() * 1000))
        cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        files = sorted(os.listdir(SAVE_DIR))
        if len(files) > MAX_IMAGES:
            for old in files[:len(files) - MAX_IMAGES]:
                try:
                    os.remove(os.path.join(SAVE_DIR, old))
                except OSError:
                    pass

    def get_stream_jpeg(self):
        """Frame moi nhat + ve box gan nhat len -> JPEG bytes (cho stream muot)."""
        with self.lock:
            if self.frame is None:
                return None
            img = self.frame.copy()
            boxes = list(self.boxes)
        draw(img, boxes)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes() if ok else None

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


WORKER = None

RESULT_HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ket qua</title><style>
body{font-family:system-ui,sans-serif;background:#0f1115;color:#e5e7eb;margin:0;padding:20px;text-align:center}
img{max-width:100%;border-radius:12px;margin:14px 0}
table{margin:10px auto;border-collapse:collapse;min-width:240px}
td{padding:6px 18px;border-bottom:1px solid #262a33;text-align:left}
a{display:inline-block;margin-top:16px;color:#4ade80;text-decoration:none;
padding:9px 24px;border:1px solid #4ade80;border-radius:22px}
h2{font-weight:600}</style></head><body>
"""
RESULT_TAIL = '<a href="/">&larr; Quay lai</a>\n</body></html>'

PAGE = """<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLOv1 - Phan loai rac</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--accent:#4ade80;--txt:#e5e7eb;--mut:#9ca3af}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}
header{padding:18px;text-align:center;border-bottom:1px solid #262a33}
header h1{font-size:20px;font-weight:600}
header .dot{color:var(--accent)}
.tabs{display:flex;justify-content:center;gap:8px;margin:18px}
.tab{padding:9px 22px;border:none;border-radius:20px;background:var(--card);
color:var(--mut);cursor:pointer;font-size:15px;transition:.2s}
.tab.on{background:var(--accent);color:#04120a;font-weight:600}
.panel{max-width:760px;margin:0 auto;padding:0 16px 40px}
.card{background:var(--card);border-radius:14px;padding:18px;margin-bottom:16px}
.view{width:100%;border-radius:10px;background:#000;display:block}
.hint{color:var(--mut);font-size:13px;margin-top:10px;text-align:center}
.up{display:flex;flex-direction:column;gap:12px;align-items:center}
input[type=file]{color:var(--mut)}
.btn{padding:10px 26px;border:none;border-radius:22px;background:var(--accent);
color:#04120a;font-weight:600;font-size:15px;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:12px}
.chip{font-size:12px;padding:3px 10px;border-radius:12px;background:#262a33;color:var(--mut)}
.hide{display:none}
</style></head><body>
<header><h1><span class="dot">&#9679;</span> YOLOv1 &mdash; Phan loai rac</h1></header>
<div class="tabs">
<button class="tab on" id="t1" onclick="show(1)">Webcam Stream</button>
<button class="tab" id="t2" onclick="show(2)">Upload anh</button>
</div>
<div class="panel">
<div class="card" id="p1">
<img class="view" src="/stream" alt="stream">
<div class="hint">Camera Jetson &bull; bounding box cap nhat theo thoi gian thuc</div>
</div>
<div class="card hide" id="p2">
<form class="up" action="/upload" method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<button class="btn" type="submit">Detect anh</button>
</form>
<div class="hint">Chon 1 anh tu may -> he thong detect va ve khung</div>
</div>
<div class="legend">
<span class="chip">BIODEGRADABLE</span><span class="chip">CARDBOARD</span>
<span class="chip">GLASS</span><span class="chip">METAL</span>
<span class="chip">PAPER</span><span class="chip">PLASTIC</span>
</div>
</div>
<script>
function show(n){
 document.getElementById('p1').classList.toggle('hide',n!=1);
 document.getElementById('p2').classList.toggle('hide',n!=2);
 document.getElementById('t1').classList.toggle('on',n==1);
 document.getElementById('t2').classList.toggle('on',n==2);
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._html(PAGE)
        elif self.path.startswith("/stream"):
            self._stream()
        elif self.path.startswith("/snapshot"):
            self._snapshot()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/upload"):
            self._upload()
        else:
            self.send_error(404)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self):
        jpg = WORKER.get_stream_jpeg()
        if jpg is None:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.end_headers()
        self.wfile.write(jpg)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        WORKER.add_viewer()               # bao "co nguoi xem" -> mo cam
        try:
            while True:
                jpg = WORKER.get_stream_jpeg()
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(
                    ("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                self.wfile.write(jpg + b"\r\n")
                time.sleep(1.0 / STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            WORKER.remove_viewer()        # het xem -> cam se tu dong dong

    def _upload(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        start = body.find(b"\r\n\r\n") + 4
        end = body.rfind(b"\r\n--")
        raw = body[start:end] if end > start else body[start:]
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self.send_error(400, "khong doc duoc anh")
            return
        boxes = WORKER.det.infer(img)
        draw(img, boxes)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        self._html(self._result_page(len(boxes), boxes, b64))

    def _result_page(self, n, boxes, b64):
        rows = ""
        for x1, y1, x2, y2, sc, c in sorted(boxes, key=lambda r: -r[4])[:20]:
            rows += ("<tr><td>%s</td><td style='color:#4ade80'>%.0f%%</td></tr>"
                     % (CLASS_NAMES[c], sc * 100))
        if not rows:
            rows = "<tr><td colspan=2 style='color:#9ca3af'>Khong tim thay vat the</td></tr>"
        return (RESULT_HEAD
                + "<h2>Tim thay " + str(n) + " vat the</h2>\n"
                + '<img src="data:image/jpeg;base64,' + b64 + '">\n'
                + "<table>" + rows + "</table>\n"
                + RESULT_TAIL)


def main():
    global WORKER
    print("Load model:", ONNX_PATH)
    det = Detector(ONNX_PATH)
    WORKER = CameraWorker(det)
    WORKER.start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("Server chay tai http://0.0.0.0:%d" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        WORKER.stop()
        srv.server_close()


if __name__ == "__main__":
    main()











