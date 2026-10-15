# -*- coding: utf-8 -*-
"""Doc dataset YOLO -> target tensor (S, S, B*5+C)."""
import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T

from config import S, B, NUM_CLASSES, IMG_SIZE


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, augment=False):
        self.img_dir = os.path.join(split_dir, "images")
        self.lbl_dir = os.path.join(split_dir, "labels")
        self.paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.augment = augment
        self.tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def _read_label(self, path):
        """Tra ve list [cls, cx, cy, w, h] (da chuan hoa 0..1)."""
        boxes = []
        if not os.path.exists(path):
            return boxes
        with open(path, "r") as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                boxes.append([c, cx, cy, w, h])
        return boxes

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, name + ".txt")

        img = Image.open(img_path).convert("RGB")
        img_t = self.tf(img)
        target = self._build_target(self._read_label(lbl_path))
        return img_t, target

    def _build_target(self, boxes):
        """Xay target (S, S, B*5+C). Moi cell: [x,y,w,h,conf]*B + one-hot lop."""
        depth = B * 5 + NUM_CLASSES
        target = torch.zeros((S, S, depth))
        for c, cx, cy, w, h in boxes:
            gi = min(int(cx * S), S - 1)   # cot cua cell
            gj = min(int(cy * S), S - 1)   # hang cua cell
            # toa do x,y tuong doi trong cell (0..1); w,h van la ty le anh
            x_cell = cx * S - gi
            y_cell = cy * S - gj
            if target[gj, gi, 4] == 1:     # cell da co object -> bo qua box nay
                continue
            for b in range(B):             # gan cho ca B box (conf=1)
                o = b * 5
                target[gj, gi, o:o + 5] = torch.tensor([x_cell, y_cell, w, h, 1.0])
            target[gj, gi, B * 5 + int(c)] = 1.0   # one-hot lop
        return target





