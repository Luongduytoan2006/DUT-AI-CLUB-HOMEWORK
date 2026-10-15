# -*- coding: utf-8 -*-
"""Train YOLOv1. Co checkpoint + auto-resume + ghi log tien do."""
import os
import time
import torch
from torch.utils.data import DataLoader

import config as C
from model import YOLOv1
from dataset import YoloDataset
from loss import YoloLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    """Ghi ra man hinh + file log (de Monitor doc duoc tien do)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    with open(C.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_ckpt(path, model, opt, epoch, best_val):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }, path)


def load_ckpt(path, model, opt):
    """Tra ve (epoch_bat_dau, best_val). Neu khong co -> (0, inf)."""
    if not os.path.exists(path):
        return 0, float("inf")
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    log(f"Resume tu epoch {ck['epoch'] + 1} (best_val={ck['best_val']:.4f})")
    return ck["epoch"] + 1, ck["best_val"]


def run_epoch(model, loader, criterion, opt, train, epoch):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, targets)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item()
        n += 1
        # log tien do trong epoch (moi 50 batch) de Monitor thay "con song"
        if train and i % 50 == 0:
            log(f"  epoch {epoch} batch {i}/{len(loader)} loss={loss.item():.4f}")
    torch.set_grad_enabled(True)
    return total / max(n, 1)


def main():
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    log(f"=== Bat dau train. Device={DEVICE} ===")

    train_ds = YoloDataset(C.TRAIN_DIR)
    valid_ds = YoloDataset(C.VALID_DIR)
    log(f"Train={len(train_ds)} anh, Valid={len(valid_ds)} anh")

    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_ld = DataLoader(valid_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                          num_workers=C.NUM_WORKERS, pin_memory=True)

    model = YOLOv1(pretrained=True).to(DEVICE)
    criterion = YoloLoss()
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    start_epoch, best_val = load_ckpt(C.CKPT_LAST, model, opt)

    for epoch in range(start_epoch, C.EPOCHS):
        t0 = time.time()
        tr_loss = run_epoch(model, train_ld, criterion, opt, True, epoch)
        va_loss = run_epoch(model, valid_ld, criterion, opt, False, epoch)
        dt = time.time() - t0
        log(f"EPOCH {epoch}/{C.EPOCHS - 1} | train={tr_loss:.4f} "
            f"val={va_loss:.4f} | {dt:.0f}s")

        save_ckpt(C.CKPT_LAST, model, opt, epoch, best_val)
        if va_loss < best_val:
            best_val = va_loss
            save_ckpt(C.CKPT_BEST, model, opt, epoch, best_val)
            log(f"  -> luu best.pt (val={best_val:.4f})")
    log("=== TRAIN XONG ===")


if __name__ == "__main__":
    main()








