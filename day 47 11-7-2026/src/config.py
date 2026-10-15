# -*- coding: utf-8 -*-
"""Cau hinh chung cho YOLOv1 - dataset rac (6 lop)."""
import os

# ------- Duong dan dataset (format YOLO: images/ + labels/) -------
DATA_ROOT = "D:/DuyToan/Python/DUT_AI/day 46 7-7-2026/data/GARBAGE CLASSIFICATION"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# ------- Lop -------
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
NUM_CLASSES = len(CLASS_NAMES)   # C = 6

# ------- Sieu tham so YOLOv1 -------
IMG_SIZE = 448      # anh vao 448x448 (chuan YOLOv1)
S = 7               # chia luoi SxS
B = 2               # so box moi o luoi

# ------- Trong so loss (theo bai bao YOLO) -------
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# ------- Train -------
BATCH_SIZE = 16
EPOCHS = 20         # train tiep len 20 epoch cho model tot hon
LR = 2e-5           # hoc nhe vi backbone pretrained
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

# ------- Luu ket qua -------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
CKPT_LAST = os.path.join(RUNS_DIR, "last.pt")     # de auto-resume
CKPT_BEST = os.path.join(RUNS_DIR, "best.pt")     # val loss tot nhat
LOG_FILE = os.path.join(RUNS_DIR, "train_log.txt")

# ------- Nguong luc suy dien -------
CONF_THRESH = 0.25
NMS_IOU = 0.45



