# -*- coding: utf-8 -*-
"""Gop JSON benchmark thanh bang Markdown de doc sau moi lan chay."""
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return {"status": "not-run", "error": "Chua co ket qua"}
    with open(path, "r") as handle:
        return json.load(handle)


def short_error(item):
    text = item.get("error", "")
    return text.replace("|", "\\|").replace("\n", " ")[:180]


rows = []
for filename, label in [("onnx.json", "ONNX"),
                        ("pt.json", "PT"),
                        ("pth.json", "PTH"),
                        ("engine.json", "TensorRT engine (tham chieu)")]:
    item = load(filename)
    rows.append("| %s | %s | %s | %s | %s |" % (
        label,
        item.get("runtime", "-"),
        item.get("status", "-"),
        item.get("pipeline_fps", item.get("inference_fps", "-")),
        short_error(item) or "-"))

report = """# Ket qua tren Jetson Nano

| Dinh dang | Runtime can dung | Trang thai | FPS pipeline | Ghi chu / loi |
|---|---|---:|---:|---|
%s

## Ket luan dung

- `.onnx` chay bang ONNX Runtime CPU trong Python 3.10, khong can PyTorch.
- `.pt` va `.pth` deu la quy uoc ten checkpoint PyTorch; duoi file khong tu tao kha nang tuong thich.
- `.pt` la checkpoint Ultralytics; `.pth` benchmark thanh cong luu full `nn.Module`. File state_dict-only rieng cho thay code kien truc phai khop chinh xac moi nap duoc.
- Kha nang chay phu thuoc CPU architecture, Python, CUDA/JetPack va runtime da cai; khong phai Windows hay Linux tu quyet dinh bang duoi file.
- TensorRT `.engine` la binary da toi uu cho dung TensorRT/GPU Jetson; day la baseline native cua NVIDIA trong bai thu.
""" % "\n".join(rows)

with open(os.path.join(RESULTS, "comparison.md"), "w") as handle:
    handle.write(report)
print(report)
