# -*- coding: utf-8 -*-
"""Xem score cao nhat tren nhieu anh test (nguong = 0)."""
import glob, os, cv2, numpy as np
from detector import Detector, CLASS_NAMES
import detector
detector.CONF_THRESH = 0.0          # lay tat ca de xem phan bo score

TEST_DIR = ("D:/DuyToan/Python/DUT_AI/day 46 7-7-2026/"
            "data/GARBAGE CLASSIFICATION/test/images")
det = Detector("../runs/yolov1_garbage.onnx")
for p in sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))[:5]:
    img = cv2.imread(p)
    res = det.infer(img)
    res = sorted(res, key=lambda r: -r[4])[:3]
    print(os.path.basename(p)[:25], "top:",
          ["%.3f %s" % (r[4], CLASS_NAMES[r[5]]) for r in res])
