# -*- coding: utf-8 -*-
"""Kiem tra nhanh: 1 batch chay qua model + loss + detect (khong train)."""
import torch
from torch.utils.data import DataLoader

import config as C
from model import YOLOv1
from dataset import YoloDataset
from loss import YoloLoss
from detect import decode

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = YoloDataset(C.TRAIN_DIR)
print("dataset:", len(ds), "anh")
ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
imgs, targets = next(iter(ld))
print("imgs", tuple(imgs.shape), "targets", tuple(targets.shape))

model = YOLOv1(pretrained=True).to(DEV)
preds = model(imgs.to(DEV))
print("preds", tuple(preds.shape))

loss = YoloLoss()(preds, targets.to(DEV))
print("loss", float(loss))

b, s, l = decode(preds[0].detach().cpu())
print("decode box0:", tuple(b.shape), "scores", tuple(s.shape))
print("=== SMOKE TEST OK ===")
