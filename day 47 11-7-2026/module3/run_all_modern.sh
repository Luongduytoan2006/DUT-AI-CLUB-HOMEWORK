#!/bin/bash
set -u
cd "$(dirname "$0")"
FRAMES="${1:-30}"

.venv-modern/bin/python src/benchmark_onnxruntime.py \
  --model models/yolov8n.onnx --video data/people.avi \
  --output results/onnx_people.avi --json results/onnx.json \
  --max-frames "$FRAMES" || true

.venv-modern/bin/python src/benchmark_pytorch.py \
  --format pt --model models/yolov8n.pt --video data/people.avi \
  --output results/pt_people.avi --json results/pt.json \
  --max-frames "$FRAMES" || true

.venv-modern/bin/python src/benchmark_pytorch.py \
  --format pth --model models/yolov8n_full_model.pth --video data/people.avi \
  --output results/pth_people.avi --json results/pth.json \
  --max-frames "$FRAMES" || true

.venv/bin/python src/benchmark_engine.py \
  --model models/yolov8n.engine --video data/people.avi \
  --output results/engine_people.avi --json results/engine.json \
  --max-frames "$FRAMES" || true

.venv-modern/bin/python src/make_report.py
