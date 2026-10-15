"""Bài 1 - Fully Convolutional Network (FCN) cho Semantic Segmentation.

File này viết theo hướng học bài:
- Phần đầu: cấu hình, tải data, đọc data.
- Phần giữa: Dataset, model FCN-32s và FCN-8s.
- Phần cuối: train loop, đánh giá, hiển thị kết quả.

Mặc định KHÔNG train. Muốn train thì chạy thêm flag --train.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ==========================================
# 1. CẤU HÌNH CHUNG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "CamVid"

IMG_SIZE = (352, 480)          # (height, width)
TRAIN_RESIZE = (384, 512)
CROP_SIZE = IMG_SIZE
BATCH_SIZE = 2
NUM_WORKERS = 0                # Windows + notebook để 0 cho dễ chạy
IGNORE_INDEX = 255
LR = 1e-4
EPOCHS = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VOID_NAMES = {"void", "unlabelled", "unlabeled", "unknown"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 2. LẤY DATA CAMVID
# ==========================================
def download_camvid():
    """Tải CamVid bằng OpenDataLab."""
    os.makedirs(BASE_DIR / "data", exist_ok=True)

    if shutil.which("odl") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opendatalab"])

    print("Nếu lần đầu dùng OpenDataLab, chạy trước trong terminal: odl login")
    subprocess.check_call(["odl", "get", "CamVid", "-d", str(BASE_DIR / "data")])
    print("Sau khi tải xong, giải nén để có folder data/CamVid gồm train, train_labels, val, val_labels, test, test_labels.")


# ==========================================
# 3. ĐỌC CLASS, MASK VÀ DATASET
# ==========================================
def doc_bang_mau(csv_path):
    """Đọc class_dict.csv: mỗi class có một màu RGB riêng."""
    class_names = []
    color_to_id = {}
    id_to_color = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            color = (int(row["r"]), int(row["g"]), int(row["b"]))

            if name.lower() in VOID_NAMES:
                color_to_id[color] = IGNORE_INDEX
            else:
                class_id = len(class_names)
                class_names.append(name)
                color_to_id[color] = class_id
                id_to_color.append(color)

    return class_names, color_to_id, id_to_color


def lay_file_bang_mau(root):
    if (root / "class_dict.csv").exists():
        return root / "class_dict.csv"
    return root / "class_palette.csv"


def lay_folder_mask(root, split):
    folder_labels = root / f"{split}_labels"
    folder_annot = root / f"{split}annot"
    if folder_labels.exists():
        return folder_labels
    return folder_annot


def doi_mask_mau_sang_mask_class(mask_rgb, color_to_id):
    """Mask gốc là ảnh màu RGB, ta đổi mỗi màu thành id của class."""
    mask_np = np.array(mask_rgb.convert("RGB"), dtype=np.uint8)
    mask_id = np.full(mask_np.shape[:2], IGNORE_INDEX, dtype=np.int64)

    for color, class_id in color_to_id.items():
        vung_mau = np.all(mask_np == np.array(color, dtype=np.uint8), axis=-1)
        mask_id[vung_mau] = class_id

    return torch.from_numpy(mask_id).long()


def bien_doi_anh_va_mask(image, mask, train=True):
    """Resize/crop/flip phải làm giống nhau cho ảnh và mask."""
    if train:
        image = TF.resize(image, TRAIN_RESIZE, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, TRAIN_RESIZE, interpolation=InterpolationMode.NEAREST)

        i, j, h, w = T.RandomCrop.get_params(image, output_size=CROP_SIZE)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
    else:
        image = TF.resize(image, IMG_SIZE, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, IMG_SIZE, interpolation=InterpolationMode.NEAREST)

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image, mask


class CamVidDataset(Dataset):
    """Đọc cặp ảnh gốc và mask nhãn của CamVid."""

    def __init__(self, root, split="train", train=True):
        self.root = Path(root)
        self.split = split
        self.train = train

        self.image_dir = self.root / split
        self.mask_dir = lay_folder_mask(self.root, split)
        self.image_paths = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.png")) + list(self.mask_dir.glob("*.jpg")))

        self.class_names, self.color_to_id, self.id_to_color = doc_bang_mau(lay_file_bang_mau(self.root))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_rgb = Image.open(self.mask_paths[idx]).convert("RGB")

        image, mask_rgb = bien_doi_anh_va_mask(image, mask_rgb, train=self.train)
        mask = doi_mask_mau_sang_mask_class(mask_rgb, self.color_to_id)
        return image, mask


def tao_dataloader():
    train_ds = CamVidDataset(DATA_ROOT, "train", train=True)
    val_ds = CamVidDataset(DATA_ROOT, "val", train=False)
    test_ds = CamVidDataset(DATA_ROOT, "test", train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_ds.class_names, train_ds.id_to_color


# ==========================================
# 4. CÁC HÀM CNN NHỎ DÙNG TRONG FCN
# ==========================================
def lay_vgg16_backbone(pretrained=False):
    """Lấy VGG16 và cắt thành 3 mức đặc trưng: pool3, pool4, pool5."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    vgg = models.vgg16(weights=weights).features

    pool3 = vgg[:17]       # stride 8,  kênh 256
    pool4 = vgg[17:24]     # stride 16, kênh 512
    pool5 = vgg[24:31]     # stride 32, kênh 512
    return pool3, pool4, pool5


def conv_1x1_score(in_channels, num_classes):
    """Conv 1x1 đổi số kênh thành số class."""
    layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def khoi_tao_upsample_bilinear(layer):
    """Khởi tạo ConvTranspose2d giống nội suy bilinear."""
    k = layer.kernel_size[0]
    factor = (k + 1) // 2
    center = factor - 1 if k % 2 == 1 else factor - 0.5
    og = np.ogrid[:k, :k]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    weight = np.zeros(layer.weight.shape, dtype=np.float32)
    for c in range(layer.in_channels):
        weight[c, c, :, :] = filt
    layer.weight.data.copy_(torch.from_numpy(weight))


def upsample_layer(num_classes, kernel_size, stride, padding):
    layer = nn.ConvTranspose2d(
        num_classes,
        num_classes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    khoi_tao_upsample_bilinear(layer)
    return layer


def chinh_ve_dung_size(x, size):
    """Đảm bảo output có đúng H,W của ảnh gốc."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ==========================================
# 5. MODEL FCN-32s VÀ FCN-8s
# ==========================================
class FCN32s(nn.Module):
    """FCN-32s: lấy pool5 rồi phóng to trực tiếp 32 lần."""

    def __init__(self, num_classes, pretrained_backbone=False):
        super().__init__()
        self.pool3, self.pool4, self.pool5 = lay_vgg16_backbone(pretrained_backbone)
        self.score_pool5 = conv_1x1_score(512, num_classes)
        self.up32 = upsample_layer(num_classes, kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        input_size = x.shape[-2:]
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)

        score = self.score_pool5(p5)
        out = self.up32(score)
        return chinh_ve_dung_size(out, input_size)


class FCN8s(nn.Module):
    """FCN-8s: cộng skip connection từ pool4 và pool3 để mask sắc hơn."""

    def __init__(self, num_classes, pretrained_backbone=False):
        super().__init__()
        self.pool3, self.pool4, self.pool5 = lay_vgg16_backbone(pretrained_backbone)

        self.score_pool3 = conv_1x1_score(256, num_classes)
        self.score_pool4 = conv_1x1_score(512, num_classes)
        self.score_pool5 = conv_1x1_score(512, num_classes)

        self.up2_pool5 = upsample_layer(num_classes, kernel_size=4, stride=2, padding=1)
        self.up2_pool4 = upsample_layer(num_classes, kernel_size=4, stride=2, padding=1)
        self.up8 = upsample_layer(num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        input_size = x.shape[-2:]
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)

        s5 = self.score_pool5(p5)
        s4 = self.score_pool4(p4)
        s3 = self.score_pool3(p3)

        up5 = self.up2_pool5(s5)
        up5 = chinh_ve_dung_size(up5, s4.shape[-2:])
        fuse4 = up5 + s4

        up4 = self.up2_pool4(fuse4)
        up4 = chinh_ve_dung_size(up4, s3.shape[-2:])
        fuse3 = up4 + s3

        out = self.up8(fuse3)
        return chinh_ve_dung_size(out, input_size)


# ==========================================
# 6. LOSS, METRIC, TRAIN LOOP
# ==========================================
def cap_nhat_confusion_matrix(confusion, preds, masks, num_classes):
    preds = preds.detach().view(-1).cpu()
    masks = masks.detach().view(-1).cpu()

    valid = masks != IGNORE_INDEX
    preds = preds[valid]
    masks = masks[valid]

    encoded = masks * num_classes + preds
    hist = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += hist.reshape(num_classes, num_classes)
    return confusion


def tinh_pa_miou(confusion):
    confusion = confusion.float()
    dung = torch.diag(confusion)
    tong = confusion.sum()

    pixel_acc = dung.sum() / tong.clamp(min=1.0)
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - dung
    iou = dung / union.clamp(min=1.0)
    miou = iou[union > 0].mean()

    return {
        "pixel_acc": float(pixel_acc.item()),
        "miou": float(miou.item()),
        "class_iou": iou.numpy(),
    }


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        confusion = cap_nhat_confusion_matrix(confusion, preds, masks, num_classes)

    metrics = tinh_pa_miou(confusion)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(model, train_loader, val_loader, num_classes, epochs=EPOCHS):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pixel_acc": [],
        "val_miou": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(val_metrics["miou"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_pixel_acc"].append(val_metrics["pixel_acc"])
        history["val_miou"].append(val_metrics["miou"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"PA={val_metrics['pixel_acc']:.4f} | "
            f"mIoU={val_metrics['miou']:.4f}"
        )

    return history


# ==========================================
# 7. HIỂN THỊ KẾT QUẢ
# ==========================================
def anh_ve_rgb(image_tensor):
    """Đổi tensor đã normalize về ảnh RGB để matplotlib hiển thị."""
    image = image_tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def mask_ve_rgb(mask, id_to_color):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(id_to_color):
        rgb[mask == class_id] = np.array(color, dtype=np.uint8)

    return rgb


@torch.no_grad()
def du_doan_1_anh(model, image):
    model.eval()
    logits = model(image.unsqueeze(0).to(DEVICE))
    pred = torch.argmax(logits, dim=1)[0]
    return pred.cpu()


@torch.no_grad()
def show_ket_qua(model32, model8, loader, id_to_color, count=5):
    model32 = model32.to(DEVICE).eval()
    model8 = model8.to(DEVICE).eval()

    images, masks = next(iter(loader))
    count = min(count, images.size(0))

    fig, axes = plt.subplots(count, 4, figsize=(14, 3.5 * count))
    if count == 1:
        axes = np.array([axes])

    for i in range(count):
        image = images[i]
        mask = masks[i]
        pred32 = du_doan_1_anh(model32, image)
        pred8 = du_doan_1_anh(model8, image)

        axes[i, 0].imshow(anh_ve_rgb(image))
        axes[i, 0].set_title("Ảnh gốc")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_ve_rgb(mask, id_to_color))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_ve_rgb(pred32, id_to_color))
        axes[i, 2].set_title("FCN-32s")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mask_ve_rgb(pred8, id_to_color))
        axes[i, 3].set_title("FCN-8s")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


def ve_history(history, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_pixel_acc"], label="val PA")
    axes[1].plot(epochs, history["val_miou"], label="val mIoU")
    axes[1].set_title("Metric")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ==========================================
# 8. MAIN - MẶC ĐỊNH KHÔNG TRAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", choices=["fcn32", "fcn8", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if args.download:
        download_camvid()
        return

    print("Device:", DEVICE)
    print("Data root:", DATA_ROOT)

    if not DATA_ROOT.exists():
        print("Chưa có data. Tải bằng lệnh:")
        print('uv run python "day 50 8-8-2026/1.py" --download')
        return

    train_loader, val_loader, test_loader, class_names, id_to_color = tao_dataloader()
    num_classes = len(class_names)

    print("Số class:", num_classes)
    print("Số batch train/val/test:", len(train_loader), len(val_loader), len(test_loader))

    model32 = FCN32s(num_classes, pretrained_backbone=False)
    model8 = FCN8s(num_classes, pretrained_backbone=False)

    x_fake = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1])
    with torch.no_grad():
        print("Output FCN-32s:", model32(x_fake).shape)
        print("Output FCN-8s:", model8(x_fake).shape)

    if not args.train:
        print("Mặc định không train. Muốn train thì thêm --train.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    if args.model in ["fcn32", "both"]:
        history32 = train_model(model32, train_loader, val_loader, num_classes, epochs=args.epochs)
        metrics32 = evaluate(model32, test_loader, criterion, num_classes)
        print("Kết quả test FCN-32s:", metrics32)
        torch.save(model32.state_dict(), BASE_DIR / "fcn32s_camvid.pt")
        ve_history(history32, "FCN-32s")

    if args.model in ["fcn8", "both"]:
        history8 = train_model(model8, train_loader, val_loader, num_classes, epochs=args.epochs)
        metrics8 = evaluate(model8, test_loader, criterion, num_classes)
        print("Kết quả test FCN-8s:", metrics8)
        torch.save(model8.state_dict(), BASE_DIR / "fcn8s_camvid.pt")
        ve_history(history8, "FCN-8s")

    show_ket_qua(model32, model8, test_loader, id_to_color, count=5)


if __name__ == "__main__":
    main()
