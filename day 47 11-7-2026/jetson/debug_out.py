# -*- coding: utf-8 -*-
"""Xem gia tri conf/score raw model xuat ra de chinh nguong."""
import glob, os, cv2, numpy as np
from detector import S, B, C, IMG_SIZE

TEST_DIR = ("D:/DuyToan/Python/DUT_AI/day 46 7-7-2026/"
            "data/GARBAGE CLASSIFICATION/test/images")
net = cv2.dnn.readNetFromONNX("../runs/yolov1_garbage.onnx")
p = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))[0]
img = cv2.imread(p)
blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
net.setInput(blob)
out = np.array(net.forward()).reshape(S, S, B*5+C)
print("out shape", out.shape)
confs = out[:, :, [4, 9]]           # 2 conf channel
print("conf min/max/mean:", confs.min(), confs.max(), confs.mean())
cls = out[:, :, B*5:]
print("cls min/max:", cls.min(), cls.max())
print("cell conf cao nhat:", np.unravel_index(confs.argmax(), confs.shape), confs.max())
