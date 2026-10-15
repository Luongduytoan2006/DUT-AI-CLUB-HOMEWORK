# -*- coding: utf-8 -*-
"""Test detector tren 1 anh rac that (chay duoi PC de kiem tra ONNX)."""
import os
import glob
import cv2
from detector import Detector, draw, CLASS_NAMES

TEST_DIR = ("D:/DuyToan/Python/DUT_AI/day 46 7-7-2026/"
            "data/GARBAGE CLASSIFICATION/test/images")
det = Detector("../runs/yolov1_garbage.onnx")

imgs = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))[:5]
for p in imgs:
    img = cv2.imread(p)
    res = det.infer(img)
    print(os.path.basename(p), "->", len(res), "box")
    for r in res[:8]:
        print("   %.2f %s" % (r[4], CLASS_NAMES[r[5]]))
    draw(img, res)
    cv2.imwrite("../test_%s" % os.path.basename(p), img)
print("DONE")
