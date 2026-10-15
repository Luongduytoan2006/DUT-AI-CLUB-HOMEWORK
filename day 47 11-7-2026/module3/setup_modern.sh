#!/bin/bash
# Chay sau khi da co Python 3.10 aarch64 tai .python/cpython-3.10.20.
set -e
cd "$(dirname "$0")"
UV=/home/team2/.local/bin/uv
PYTHON=.python/cpython-3.10.20/bin/python

"$UV" venv .venv-modern --python "$PYTHON"
"$UV" pip install --python .venv-modern/bin/python -r requirements-modern.txt
# Cai no-deps de giu opencv-python-headless, tranh cai them opencv-python GUI.
"$UV" pip install --python .venv-modern/bin/python --no-deps ultralytics==8.3.0

.venv-modern/bin/python -c \
  'import torch, cv2, onnxruntime, ultralytics; print(torch.__version__, cv2.__version__, onnxruntime.__version__, ultralytics.__version__)'

