# -*- coding: utf-8 -*-
"""YOLOv1 voi backbone ResNet18 (nhe, hop Jetson Nano)."""
import torch
import torch.nn as nn
import torchvision.models as models

from config import S, B, NUM_CLASSES


class YOLOv1(nn.Module):
    """Dau ra: (N, S, S, B*5 + C). Moi box = [x, y, w, h, conf]."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.S, self.B, self.C = S, B, NUM_CLASSES
        out_dim = B * 5 + NUM_CLASSES

        # Backbone ResNet18, bo avgpool + fc -> feature map (512, 14, 14) voi anh 448
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Dau detection: 14x14 -> 7x7 roi conv ra out_dim kenh
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, out_dim, 1),                    # 1x1 -> (B*5+C)
        )

    def forward(self, x):
        x = self.features(x)          # (N, 512, 14, 14)
        x = self.head(x)              # (N, out_dim, 7, 7)
        x = x.permute(0, 2, 3, 1)     # (N, 7, 7, out_dim)
        return x.contiguous()




