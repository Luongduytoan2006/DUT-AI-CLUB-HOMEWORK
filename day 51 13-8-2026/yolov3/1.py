"""B?i YOLOv3 from scratch cho Road Damage Detection.

File n?y vi?t theo h??ng h?c b?i gi?ng day 50:
- Khai b?o c?u h?nh, h?m ??c data, Dataset ? ??u file.
- Khai b?o class ki?n tr?c model, loss, train/evaluate ? gi?a file.
- Cu?i file ch? c? main() g?i solve().

M?c ??nh KH?NG train. Mu?n train th? ch?y th?m flag --train.
"""


import math
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms, box_iou
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    display
except NameError:
    display = print


# =========================================================
# CẤU HÌNH
# =========================================================
IN_COLAB = "COLAB_RELEASE_TAG" in __import__("os").environ
BASE_DIR = Path(__file__).resolve().parent
IN_KAGGLE = Path("/kaggle/working").exists()

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DATA_ROOT = Path("/content/drive/MyDrive/datasets/road_damage_pcm")
    except (ImportError, RuntimeError):
        DATA_ROOT = Path("/content/data/road_damage_pcm")
elif IN_KAGGLE:
    DATA_ROOT = Path("/kaggle/working/datasets/road_damage_pcm")
else:
    DATA_ROOT = BASE_DIR / "data" / "road_damage_pcm"

IMAGE_SIZE = 416
BATCH_SIZE = 2
NUM_CLASSES = 3
CLASS_NAMES = ["Pothole", "Crack", "Manhole"]
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTO_DOWNLOAD_DATASET = False

STRIDES = (8, 16, 32)
ANCHORS_PER_SCALE = 3
PREDICTION_SIZE = 5 + NUM_CLASSES
DEFAULT_ANCHORS = torch.tensor([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326],
], dtype=torch.float32)

assert IMAGE_SIZE % 32 == 0, "IMAGE_SIZE phải chia hết cho 32."



ANCHORS = DEFAULT_ANCHORS.clone()
ANCHORS_BY_SCALE = ANCHORS.view(3, 3, 2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# TẢI VÀ CHUẨN HÓA ROAD DAMAGE DATASET
# =========================================================
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
)
DATASET_ARCHIVE = DATA_ROOT.parent / "road_damage_pcm.zip"


def download_file(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def prepare_road_damage_dataset(data_root=DATA_ROOT):
    ready_file = data_root / ".ready"
    if ready_file.exists():
        print(f"Dataset đã sẵn sàng tại: {data_root}")
        return

    DATASET_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_ARCHIVE.exists() or not zipfile.is_zipfile(DATASET_ARCHIVE):
        print("Đang tải Road Damage Dataset khoảng 195 MB...")
        download_file(DATASET_URL, DATASET_ARCHIVE)

    image_prefix = "data/images/"
    label_prefix = "data/labels-YOLO/"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with zipfile.ZipFile(DATASET_ARCHIVE) as archive:
        names = archive.namelist()
        labels = set(name for name in names if name.startswith(label_prefix))
        pairs = []
        for image_name in names:
            if not image_name.startswith(image_prefix) or Path(image_name).suffix.lower() not in image_exts:
                continue
            label_name = f"{label_prefix}{Path(image_name).stem}.txt"
            if label_name in labels:
                pairs.append((image_name, label_name))

        random.Random(SEED).shuffle(pairs)
        split_index = int(0.8 * len(pairs))
        for index, (image_name, label_name) in enumerate(tqdm(pairs, desc="Chuẩn hóa dataset")):
            split = "train" if index < split_index else "valid"
            for member, folder in ((image_name, "images"), (label_name, "labels")):
                destination = data_root / split / folder / Path(member).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

    data_yaml = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": CLASS_NAMES,
    }
    with (data_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False, allow_unicode=True)
    ready_file.write_text(str(len(pairs)), encoding="utf-8")
    print(f"Hoàn tất: {split_index} train | {len(pairs) - split_index} validation")


# =========================================================
# DATASET VÀ DATALOADER
# =========================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODetectionDataset(Dataset):
    def __init__(self, data_root, split, image_size):
        self.image_dir = Path(data_root) / split / "images"
        self.label_dir = Path(data_root) / split / "labels"
        self.image_size = image_size
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy {self.image_dir}.")
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = TF.to_tensor(TF.resize(image, [self.image_size, self.image_size], antialias=True))

        labels, boxes = [], []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x, y, width, height = line.split()
                labels.append(int(class_id))
                boxes.append([float(x), float(y), float(width), float(height)])

        return image, {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_dataloaders():
    train_dataset = YOLODetectionDataset(DATA_ROOT, "train", IMAGE_SIZE)
    val_dataset = YOLODetectionDataset(DATA_ROOT, "valid", IMAGE_SIZE)
    options = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    return train_dataset, val_dataset, train_loader, val_loader


# =========================================================
# VISUALIZE DATASET
# =========================================================
BOX_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7"]


def xywhn_to_xyxy(boxes, image_size):
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([
        (x - w / 2) * image_size,
        (y - h / 2) * image_size,
        (x + w / 2) * image_size,
        (y + h / 2) * image_size,
    ], dim=-1)


def draw_boxes(image, boxes, labels, scores=None, class_names=CLASS_NAMES):
    canvas = TF.to_pil_image(image.cpu().clamp(0, 1))
    draw = ImageDraw.Draw(canvas)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        color = BOX_COLORS[int(label) % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = class_names[int(label)]
        if scores is not None:
            text += f" {float(scores[index]):.2f}"
        draw.text((x1 + 3, max(0, y1 - 12)), text, fill=color)
    return canvas


def show_dataset_samples(dataset, count=3):
    count = min(count, len(dataset))
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for index in range(count):
        image, target = dataset[index]
        boxes = xywhn_to_xyxy(target["boxes"], image.shape[-1])
        axes[0, index].imshow(draw_boxes(image, boxes, target["labels"]))
        axes[0, index].set_title(f"Ground truth #{index}")
        axes[0, index].axis("off")
    plt.tight_layout()
    display(fig)
    plt.close(fig)


# =========================================================
# K-MEANS ANCHOR BOXES
# =========================================================
def wh_iou(box_wh, cluster_wh):
    box_wh = box_wh[:, None, :]
    cluster_wh = cluster_wh[None, :, :]
    intersection = torch.minimum(box_wh, cluster_wh).prod(dim=-1)
    union = box_wh.prod(dim=-1) + cluster_wh.prod(dim=-1) - intersection
    return intersection / (union + 1e-9)


def collect_box_wh(dataset):
    all_wh = []
    for image_path in tqdm(dataset.image_paths, desc="Đọc label để cluster anchor"):
        label_path = dataset.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, _, _, width, height = line.split()
            all_wh.append([float(width) * IMAGE_SIZE, float(height) * IMAGE_SIZE])
    if not all_wh:
        return torch.empty(0, 2)
    return torch.tensor(all_wh, dtype=torch.float32).clamp(min=1.0)


def kmeans_anchors(box_wh, k=9, iterations=50, seed=SEED):
    if box_wh.shape[0] < k:
        return DEFAULT_ANCHORS.clone()

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(box_wh.shape[0], generator=generator)[:k]
    clusters = box_wh[indices].clone()

    for _ in range(iterations):
        distances = 1 - wh_iou(box_wh, clusters)
        assignments = distances.argmin(dim=1)
        new_clusters = clusters.clone()
        for cluster_index in range(k):
            members = box_wh[assignments == cluster_index]
            if len(members) > 0:
                new_clusters[cluster_index] = members.median(dim=0).values
        if torch.allclose(clusters, new_clusters):
            break
        clusters = new_clusters

    areas = clusters.prod(dim=1)
    return clusters[areas.argsort()]


# =========================================================
# YOLOV3 BUILDING BLOCKS
# =========================================================
class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels // 2
        self.conv1 = ConvBNLeaky(channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNLeaky(hidden_channels, channels, kernel_size=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


def make_residual_stage(in_channels, out_channels, repeats):
    layers = [ConvBNLeaky(in_channels, out_channels, kernel_size=3, stride=2)]
    layers += [ResidualBlock(out_channels) for _ in range(repeats)]
    return nn.Sequential(*layers)


# =========================================================
# BACKBONE: DARKNET-53
# =========================================================
class Darknet53Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNLeaky(3, 32, kernel_size=3, stride=1)
        self.stage1 = make_residual_stage(32, 64, repeats=1)
        self.stage2 = make_residual_stage(64, 128, repeats=2)
        self.stage3 = make_residual_stage(128, 256, repeats=8)
        self.stage4 = make_residual_stage(256, 512, repeats=8)
        self.stage5 = make_residual_stage(512, 1024, repeats=4)
        self.out_channels = (256, 512, 1024)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        small = self.stage3(x)
        medium = self.stage4(small)
        large = self.stage5(medium)
        return small, medium, large


# =========================================================
# NECK: FPN TOP-DOWN
# =========================================================
class YOLOv3ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNLeaky(in_channels, out_channels, kernel_size=1),
            ConvBNLeaky(out_channels, out_channels * 2, kernel_size=3),
            ConvBNLeaky(out_channels * 2, out_channels, kernel_size=1),
            ConvBNLeaky(out_channels, out_channels * 2, kernel_size=3),
            ConvBNLeaky(out_channels * 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class YOLOv3Neck(nn.Module):
    def __init__(self, channels=(256, 512, 1024)):
        super().__init__()
        c_small, c_medium, c_large = channels
        self.large_set = YOLOv3ConvSet(c_large, 512)
        self.large_route = ConvBNLeaky(512, 256, kernel_size=1)
        self.medium_set = YOLOv3ConvSet(256 + c_medium, 256)
        self.medium_route = ConvBNLeaky(256, 128, kernel_size=1)
        self.small_set = YOLOv3ConvSet(128 + c_small, 128)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_channels = (128, 256, 512)

    def forward(self, features):
        small, medium, large = features
        large_feature = self.large_set(large)
        medium_input = torch.cat([self.upsample(self.large_route(large_feature)), medium], dim=1)
        medium_feature = self.medium_set(medium_input)
        small_input = torch.cat([self.upsample(self.medium_route(medium_feature)), small], dim=1)
        small_feature = self.small_set(small_input)
        return small_feature, medium_feature, large_feature


# =========================================================
# DETECTION HEAD
# =========================================================
class YOLOv3DetectionHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = ANCHORS_PER_SCALE * (5 + num_classes)
        self.heads = nn.ModuleList()
        for channel in channels:
            self.heads.append(nn.Sequential(
                ConvBNLeaky(channel, channel * 2, kernel_size=3),
                nn.Conv2d(channel * 2, self.out_channels, kernel_size=1),
            ))

    def forward(self, features):
        return [head(feature) for head, feature in zip(self.heads, features)]


class YOLOv3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Darknet53Backbone()
        self.neck = YOLOv3Neck(self.backbone.out_channels)
        self.head = YOLOv3DetectionHead(self.neck.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)


# =========================================================
# TARGET ASSIGNMENT CHO YOLOV3
# =========================================================
def reshape_predictions(output, num_classes=NUM_CLASSES):
    batch_size, _, grid_h, grid_w = output.shape
    output = output.view(batch_size, ANCHORS_PER_SCALE, 5 + num_classes, grid_h, grid_w)
    return output.permute(0, 1, 3, 4, 2).contiguous()


def xywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat([center - size / 2, center + size / 2], dim=-1)


def build_yolov3_targets(outputs, targets, anchors_by_scale, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    device = outputs[0].device
    all_anchors = anchors_by_scale.reshape(-1, 2).to(device)
    scale_targets = []

    for output in outputs:
        batch_size, _, grid_h, grid_w = output.shape
        target = torch.zeros(
            batch_size, ANCHORS_PER_SCALE, grid_h, grid_w, 5 + num_classes,
            device=device,
        )
        scale_targets.append(target)

    for batch_index, target in enumerate(targets):
        labels = target["labels"].to(device)
        boxes = target["boxes"].to(device)
        if labels.numel() == 0:
            continue

        box_wh_pixels = boxes[:, 2:4] * image_size
        best_anchor = wh_iou(box_wh_pixels.cpu(), all_anchors.cpu()).argmax(dim=1).to(device)

        for object_index, anchor_index in enumerate(best_anchor):
            scale_index = int(anchor_index // ANCHORS_PER_SCALE)
            anchor_in_scale = int(anchor_index % ANCHORS_PER_SCALE)
            grid_size = scale_targets[scale_index].shape[2]

            cx, cy, bw, bh = boxes[object_index]
            grid_x = torch.clamp((cx * grid_size).long(), 0, grid_size - 1)
            grid_y = torch.clamp((cy * grid_size).long(), 0, grid_size - 1)
            if scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 4] == 1:
                continue

            anchor_wh = all_anchors[anchor_index]
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 0] = cx * grid_size - grid_x.float()
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 1] = cy * grid_size - grid_y.float()
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 2] = torch.log(bw * image_size / anchor_wh[0] + 1e-9)
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 3] = torch.log(bh * image_size / anchor_wh[1] + 1e-9)
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 4] = 1.0
            scale_targets[scale_index][batch_index, anchor_in_scale, grid_y, grid_x, 5 + labels[object_index]] = 1.0

    return scale_targets


# =========================================================
# YOLOV3 LOSS
# =========================================================
class YOLOv3Loss:
    def __init__(self, anchors_by_scale, num_classes=NUM_CLASSES):
        self.anchors_by_scale = anchors_by_scale.float()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def __call__(self, outputs, targets, image_size=IMAGE_SIZE):
        target_maps = build_yolov3_targets(
            outputs, targets, self.anchors_by_scale, image_size, self.num_classes
        )
        total_xy = total_wh = total_obj = total_cls = 0.0
        positive_count = 0.0

        for output, target in zip(outputs, target_maps):
            pred = reshape_predictions(output, self.num_classes)
            obj_mask = target[..., 4] == 1
            noobj_mask = ~obj_mask
            positive_count += obj_mask.sum().float()

            total_obj = total_obj + self.bce(pred[..., 4][obj_mask], target[..., 4][obj_mask])
            total_obj = total_obj + 0.5 * self.bce(pred[..., 4][noobj_mask], target[..., 4][noobj_mask])

            if obj_mask.any():
                total_xy = total_xy + self.bce(pred[..., 0:2][obj_mask], target[..., 0:2][obj_mask])
                total_wh = total_wh + self.mse(pred[..., 2:4][obj_mask], target[..., 2:4][obj_mask])
                total_cls = total_cls + self.bce(pred[..., 5:][obj_mask], target[..., 5:][obj_mask])
            else:
                total_xy = total_xy + output.sum() * 0.0
                total_wh = total_wh + output.sum() * 0.0
                total_cls = total_cls + output.sum() * 0.0

        normalizer = positive_count.clamp(min=1.0)
        box_loss = (total_xy + total_wh) / normalizer
        obj_loss = total_obj / normalizer
        cls_loss = total_cls / normalizer
        loss = box_loss + obj_loss + cls_loss
        parts = {
            "loss": loss.detach(),
            "box_loss": box_loss.detach(),
            "obj_loss": obj_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "positive": positive_count.detach(),
        }
        return loss, parts


def move_targets_to_device(targets, device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"loss": 0.0, "box_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0}
    total_items = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            targets = move_targets_to_device(targets, DEVICE)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss, parts = criterion(outputs, targets, image_size=images.shape[-1])
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_items += batch_size
            for key in totals:
                totals[key] += float(parts[key]) * batch_size

    return {key: value / max(total_items, 1) for key, value in totals.items()}


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    model = model.to(DEVICE)
    criterion = YOLOv3Loss(ANCHORS_BY_SCALE, NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_stats = run_epoch(model, train_loader, criterion, optimizer)
        val_stats = run_epoch(model, val_loader, criterion)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train {train_stats['loss']:.4f} | val {val_stats['loss']:.4f} | "
            f"box {val_stats['box_loss']:.4f} | obj {val_stats['obj_loss']:.4f} | "
            f"cls {val_stats['cls_loss']:.4f} | {time.time() - start:.1f}s"
        )
    return model, pd.DataFrame(history)


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="validation")
    for name in ("box_loss", "obj_loss", "cls_loss"):
        axes[1].plot(history["epoch"], history[f"val_{name}"], marker="o", label=name)
    axes[0].set(title="Total loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation loss components", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    plt.tight_layout()
    display(fig)
    plt.close(fig)


# =========================================================
# DECODE OUTPUT + NMS
# =========================================================
def decode_yolov3_outputs(outputs, anchors_by_scale, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    all_boxes, all_scores, all_labels = [], [], []
    device = outputs[0].device
    anchors_by_scale = anchors_by_scale.to(device)

    for scale_index, output in enumerate(outputs):
        pred = reshape_predictions(output, num_classes)
        batch_size, _, grid_h, grid_w, _ = pred.shape
        stride = image_size / grid_h
        anchors = anchors_by_scale[scale_index].view(1, ANCHORS_PER_SCALE, 1, 1, 2)

        y, x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=pred.dtype),
            torch.arange(grid_w, device=device, dtype=pred.dtype),
            indexing="ij",
        )
        grid = torch.stack([x, y], dim=-1).view(1, 1, grid_h, grid_w, 2)

        center = (pred[..., 0:2].sigmoid() + grid) * stride
        size = anchors * pred[..., 2:4].exp()
        boxes = torch.cat([center - size / 2, center + size / 2], dim=-1)
        objectness = pred[..., 4].sigmoid()
        class_prob = pred[..., 5:].sigmoid()
        class_scores, labels = class_prob.max(dim=-1)
        scores = objectness * class_scores

        all_boxes.append(boxes.reshape(batch_size, -1, 4))
        all_scores.append(scores.reshape(batch_size, -1))
        all_labels.append(labels.reshape(batch_size, -1))

    return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1), torch.cat(all_labels, dim=1)


@torch.no_grad()
def postprocess(outputs, confidence_threshold=0.25, iou_threshold=0.5, max_detections=100):
    boxes, scores, labels = decode_yolov3_outputs(outputs, ANCHORS_BY_SCALE, IMAGE_SIZE, NUM_CLASSES)
    results = []
    for image_boxes, image_scores, image_labels in zip(boxes, scores, labels):
        keep = image_scores >= confidence_threshold
        image_boxes = image_boxes[keep]
        image_scores = image_scores[keep]
        image_labels = image_labels[keep]
        if image_boxes.numel() == 0:
            results.append({"boxes": image_boxes.reshape(0, 4), "scores": image_scores, "labels": image_labels})
            continue
        keep_indices = batched_nms(image_boxes, image_scores, image_labels, iou_threshold)
        keep_indices = keep_indices[:max_detections]
        results.append({
            "boxes": image_boxes[keep_indices],
            "scores": image_scores[keep_indices],
            "labels": image_labels[keep_indices],
        })
    return results


@torch.no_grad()
def predict_batch(model, images, confidence_threshold=0.25, iou_threshold=0.5):
    model.eval()
    outputs = model(images.to(DEVICE))
    return postprocess(outputs, confidence_threshold, iou_threshold)


def show_predictions(model, dataset, count=3, confidence_threshold=0.25):
    count = min(count, len(dataset))
    images, targets = [], []
    for index in range(count):
        image, target = dataset[index]
        images.append(image)
        targets.append(target)
    batch = torch.stack(images)
    predictions = predict_batch(model, batch, confidence_threshold=confidence_threshold)

    fig, axes = plt.subplots(count, 2, figsize=(10, 5 * count), squeeze=False)
    for index in range(count):
        gt_boxes = xywhn_to_xyxy(targets[index]["boxes"], IMAGE_SIZE)
        axes[index, 0].imshow(draw_boxes(images[index], gt_boxes, targets[index]["labels"]))
        axes[index, 0].set_title("Ground truth")
        axes[index, 0].axis("off")

        pred = predictions[index]
        axes[index, 1].imshow(draw_boxes(images[index], pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()))
        axes[index, 1].set_title("Prediction")
        axes[index, 1].axis("off")
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def run_shape_test():
    # =========================================================
    # SHAPE TEST
    # =========================================================
    set_seed(SEED)
    model = YOLOv3(NUM_CLASSES)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    model.eval()
    with torch.no_grad():
        backbone_features = model.backbone(x)
        neck_features = model.neck(backbone_features)
        outputs = model.head(neck_features)

    expected_hw = [IMAGE_SIZE // stride for stride in STRIDES]
    expected_backbone_channels = model.backbone.out_channels
    expected_neck_channels = model.neck.out_channels
    expected_output_channels = ANCHORS_PER_SCALE * (5 + NUM_CLASSES)

    for index, (bf, nf, out, hw, bc, nc) in enumerate(
        zip(backbone_features, neck_features, outputs, expected_hw, expected_backbone_channels, expected_neck_channels),
        start=1,
    ):
        assert bf.shape == (2, bc, hw, hw), f"Backbone scale {index} sai shape: {bf.shape}"
        assert nf.shape == (2, nc, hw, hw), f"Neck scale {index} sai shape: {nf.shape}"
        assert out.shape == (2, expected_output_channels, hw, hw), f"Head scale {index} sai shape: {out.shape}"
        print(f"Scale {index}: backbone={tuple(bf.shape)} | neck={tuple(nf.shape)} | pred={tuple(out.shape)}")

    full_outputs = model(x)
    assert len(full_outputs) == 3, "YOLOv3 phải trả về 3 scale."
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Full forward: OK | Parameters: {parameter_count / 1e6:.2f}M")



def run_smoke_test():
    # =========================================================
    # LOSS + BACKWARD SMOKE TEST
    # =========================================================
    set_seed(SEED)
    smoke_model = YOLOv3(NUM_CLASSES).to(DEVICE).train()
    smoke_images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    smoke_targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.50, 0.50, 0.35, 0.30]])},
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.35, 0.40, 0.25, 0.20]])},
    ]
    smoke_outputs = smoke_model(smoke_images)
    smoke_criterion = YOLOv3Loss(ANCHORS_BY_SCALE, NUM_CLASSES)
    smoke_loss, smoke_parts = smoke_criterion(smoke_outputs, smoke_targets, image_size=IMAGE_SIZE)
    assert torch.isfinite(smoke_loss), f"Loss không hữu hạn: {smoke_loss}"
    assert smoke_parts["positive"].item() > 0, "Target assignment chưa tạo positive anchor."
    smoke_loss.backward()
    has_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in smoke_model.parameters()
    )
    assert has_gradient, "Backward chưa tạo gradient hữu hạn."
    print({key: float(value) for key, value in smoke_parts.items()})
    print("Loss + backward: OK")
    del smoke_model, smoke_images, smoke_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




def build_or_load_data(auto_download=False):
    if auto_download:
        try:
            prepare_road_damage_dataset()
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
            print("Kh?ng th? t?i dataset:", error)

    try:
        train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()
        print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
        return True, train_dataset, val_dataset, train_loader, val_loader
    except FileNotFoundError as error:
        print(error)
        print("Ch?a c? dataset. C? th? ch?y th?m --download n?u mu?n t?i/chu?n h?a data.")
        return False, None, None, None, None


def solve(args):
    global ANCHORS, ANCHORS_BY_SCALE

    set_seed(SEED)
    print("Device:", DEVICE)
    print("Data root:", DATA_ROOT)

    data_available, train_dataset, val_dataset, train_loader, val_loader = build_or_load_data(
        auto_download=args.download or AUTO_DOWNLOAD_DATASET
    )

    if data_available:
        box_wh = collect_box_wh(train_dataset)
        ANCHORS = kmeans_anchors(box_wh)
        ANCHORS_BY_SCALE = ANCHORS.view(3, 3, 2)
        mean_iou = wh_iou(box_wh, ANCHORS).max(dim=1).values.mean().item() if len(box_wh) else 0.0
        print("K-Means anchors:")
        print(ANCHORS.round().int())
        print(f"Mean best IoU: {mean_iou:.3f}")
    else:
        ANCHORS = DEFAULT_ANCHORS.clone()
        ANCHORS_BY_SCALE = ANCHORS.view(3, 3, 2)
        print("D?ng default YOLOv3 anchors v? dataset ch?a c?:")
        print(ANCHORS.int())

    print("Anchors theo scale 52/26/13:")
    print(ANCHORS_BY_SCALE.round().int())

    run_shape_test()
    run_smoke_test()

    if not args.train:
        print("M?c ??nh kh?ng train. Mu?n train th? th?m --train.")
        return

    if not data_available:
        print("Kh?ng train v? dataset ch?a s?n s?ng.")
        return

    model = YOLOv3(NUM_CLASSES)
    model, history = train_model(model, train_loader, val_loader, args.epochs)
    display(history.style.format(precision=4))
    plot_history(history)

    if args.visualize:
        show_dataset_samples(train_dataset)
        show_predictions(model, val_dataset, count=3, confidence_threshold=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="T?i/chu?n h?a dataset n?u ch?a c?.")
    parser.add_argument("--train", action="store_true", help="Train model. M?c ??nh ch? ch?y shape/smoke test.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--visualize", action="store_true", help="Hi?n th? sample/prediction sau train.")
    args = parser.parse_args()
    solve(args)


if __name__ == "__main__":
    main()
