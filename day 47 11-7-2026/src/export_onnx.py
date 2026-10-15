# -*- coding: utf-8 -*-
"""Export model .pt -> .onnx de chay tren Jetson bang OpenCV DNN (khong can torch)."""
import os
import torch

import config as C
from model import YOLOv1

ONNX_PATH = os.path.join(C.RUNS_DIR, "yolov1_garbage.onnx")


def export(ckpt_path=C.CKPT_BEST):
    model = YOLOv1(pretrained=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    print("Loaded", ckpt_path, "epoch", ck.get("epoch"))

    dummy = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["images"], output_names=["pred"],
        opset_version=11,               # opset 11 -> tuong thich OpenCV 4.1 tren Jetson
        dynamic_axes=None,              # co dinh batch=1 cho on dinh
    )
    print("Da xuat ONNX:", ONNX_PATH)


if __name__ == "__main__":
    export()


