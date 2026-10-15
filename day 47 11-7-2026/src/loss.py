# -*- coding: utf-8 -*-
"""YOLOv1 loss (theo bai bao goc)."""
import torch
import torch.nn as nn

from config import S, B, NUM_CLASSES, LAMBDA_COORD, LAMBDA_NOOBJ


def _xywh_to_xyxy(box):
    """box (...,4) dang [x,y,w,h] -> [x1,y1,x2,y2]. x,y la tam."""
    x1 = box[..., 0] - box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _iou(box1, box2):
    """IoU giua 2 box [x,y,w,h] cung shape (...,4)."""
    b1 = _xywh_to_xyxy(box1)
    b2 = _xywh_to_xyxy(box2)
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (b1[..., 2] - b1[..., 0]).clamp(0) * (b1[..., 3] - b1[..., 1]).clamp(0)
    a2 = (b2[..., 2] - b2[..., 0]).clamp(0) * (b2[..., 3] - b2[..., 1]).clamp(0)
    return inter / (a1 + a2 - inter + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        # pred, target: (N, S, S, B*5+C)
        N = pred.size(0)
        obj = target[..., 4] > 0            # (N,S,S) cell co object
        noobj = ~obj

        # IoU cua tung box du doan voi box that -> chon box trach nhiem
        gt_box = target[..., 0:4]                       # (N,S,S,4)
        ious = []
        for b in range(B):
            pb = pred[..., b * 5:b * 5 + 4]
            ious.append(_iou(pb, gt_box).unsqueeze(-1)) # (N,S,S,1)
        ious = torch.cat(ious, dim=-1)                  # (N,S,S,B)
        best_iou, best_b = ious.max(dim=-1)             # (N,S,S)

        # Gom box trach nhiem: chon theo best_b cho tung cell
        pred_boxes = torch.stack(
            [pred[..., b * 5:b * 5 + 5] for b in range(B)], dim=-2
        )                                                # (N,S,S,B,5)
        idx = best_b[..., None, None].expand(-1, -1, -1, 1, 5)
        resp = pred_boxes.gather(-2, idx).squeeze(-2)    # (N,S,S,5) box trach nhiem

        om = obj.unsqueeze(-1)                           # (N,S,S,1) mask float sau
        # --- 1) Coord loss (chi cell co object) ---
        p_xy = resp[..., 0:2]
        t_xy = target[..., 0:2]
        loss_xy = self.mse(om * p_xy, om * t_xy)
        # w,h: lay can bac 2 (giam anh huong box lon), giu dau
        p_wh = torch.sign(resp[..., 2:4]) * torch.sqrt(resp[..., 2:4].abs() + 1e-6)
        t_wh = torch.sqrt(target[..., 2:4] + 1e-6)
        loss_wh = self.mse(om * p_wh, om * t_wh)
        loss_coord = LAMBDA_COORD * (loss_xy + loss_wh)

        # --- 2) Object confidence: muc tieu = IoU cua box trach nhiem ---
        p_conf = resp[..., 4]
        loss_obj = self.mse(obj * p_conf, obj * best_iou.detach())

        # --- 3) No-object confidence (ca B box, cell khong co object) ---
        loss_noobj = 0.0
        for b in range(B):
            cb = pred[..., b * 5 + 4]
            loss_noobj += self.mse(noobj * cb, noobj * torch.zeros_like(cb))
        loss_noobj = LAMBDA_NOOBJ * loss_noobj

        # --- 4) Class loss (chi cell co object) ---
        p_cls = pred[..., B * 5:]
        t_cls = target[..., B * 5:]
        cm = obj.unsqueeze(-1)
        loss_cls = self.mse(cm * p_cls, cm * t_cls)

        total = (loss_coord + loss_obj + loss_noobj + loss_cls) / N
        return total







