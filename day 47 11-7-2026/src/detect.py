# -*- coding: utf-8 -*-
"""Giai ma output YOLOv1 -> boxes + NMS. Dung chung local & Jetson."""
import torch
from torchvision.ops import nms

from config import S, B, NUM_CLASSES, IMG_SIZE, CONF_THRESH, NMS_IOU


def decode(pred, conf_thresh=CONF_THRESH):
    """pred (S,S,B*5+C) 1 anh -> (boxes_xyxy px, scores, labels). Toa do theo IMG_SIZE."""
    boxes, scores, labels = [], [], []
    cls_prob = pred[..., B * 5:]                 # (S,S,C)
    cls_score, cls_idx = cls_prob.max(dim=-1)    # (S,S)

    cell = IMG_SIZE / S
    for gj in range(S):
        for gi in range(S):
            for b in range(B):
                o = b * 5
                conf = pred[gj, gi, o + 4].item()
                score = conf * cls_score[gj, gi].item()
                if score < conf_thresh:
                    continue
                x, y, w, h = pred[gj, gi, o:o + 4].tolist()
                # x,y tuong doi trong cell -> tam tuyet doi (px)
                cx = (gi + x) * cell
                cy = (gj + y) * cell
                bw = w * IMG_SIZE
                bh = h * IMG_SIZE
                boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                scores.append(score)
                labels.append(cls_idx[gj, gi].item())

    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels, dtype=torch.long)

    # NMS theo tung lop
    keep_all = []
    for c in labels.unique():
        m = labels == c
        idx = torch.where(m)[0]
        k = nms(boxes[m], scores[m], NMS_IOU)
        keep_all.append(idx[k])
    keep = torch.cat(keep_all)
    return boxes[keep], scores[keep], labels[keep]




