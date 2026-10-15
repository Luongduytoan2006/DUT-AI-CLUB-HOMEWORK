# -*- coding: utf-8 -*-
"""Detect 1 anh tu file. Dung: python3 detect_image.py anh.jpg [ket_qua.jpg]"""
import sys
import os
import cv2
from detector import Detector, draw, CLASS_NAMES

ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "yolov1_garbage.onnx")


def main():
    if len(sys.argv) < 2:
        print("Dung: python3 detect_image.py <anh.jpg> [out.jpg]")
        return
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "detected.jpg"
    img = cv2.imread(src)
    if img is None:
        print("Khong doc duoc anh:", src)
        return
    det = Detector(ONNX)
    res = det.infer(img)
    print("Tim thay %d vat the:" % len(res))
    for x1, y1, x2, y2, sc, c in res:
        print("  %s  score=%.2f  box=(%d,%d,%d,%d)"
              % (CLASS_NAMES[c], sc, x1, y1, x2, y2))
    draw(img, res)
    cv2.imwrite(out, img)
    print("Da luu:", out)


if __name__ == "__main__":
    main()
