# -*- coding: utf-8 -*-
"""Export yolov8n.pt (COCO) -> yolov8n.onnx de build TensorRT engine tren Jetson.
Static shape 1x3x640x640, opset 12, simplify -> on dinh nhat cho TRT 8.2.
"""
from ultralytics import YOLO


def main():
    # tai yolov8n.pt COCO pretrained (auto download lan dau)
    model = YOLO("yolov8n.pt")
    # export ONNX: static batch, opset 12, don gian hoa graph
    path = model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        simplify=True,
        dynamic=False,
    )
    print("EXPORTED:", path)


if __name__ == "__main__":
    main()
