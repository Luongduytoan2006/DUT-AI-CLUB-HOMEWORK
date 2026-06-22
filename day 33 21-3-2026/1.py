import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

SEED = 42
DATA_ROOT = "/kaggle/input/emotion-recognition-dataset/dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")
IMAGE_SIZE = 224
BATCH_SIZE = 36
NUM_EPOCHS = 18
PATIENCE = 9
NUM_WORKERS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
SAVE_DIR = "./my_feature_outputs"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

class_names = train_dataset.classes
num_classes = len(class_names)

class MyResNet18FeatureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_bn_eval(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

def my_run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
        model.set_backbone_bn_eval()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc

def my_plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def my_train_feature_extraction():
    model = MyResNet18FeatureNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    early_stop_count = 0

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = my_run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = my_run_epoch(model, val_loader, criterion, optimizer=None)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            early_stop_count = 0
            torch.save(best_state, os.path.join(SAVE_DIR, "my_resnet18_feature_best.pth"))
        else:
            early_stop_count += 1

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if early_stop_count >= PATIENCE:
            break

    total_time = time.time() - start_time

    model.load_state_dict(best_state)
    test_loss, test_acc = my_run_epoch(model, test_loader, criterion, optimizer=None)

    my_plot_history(history, os.path.join(SAVE_DIR, "my_resnet18_feature_curves.png"))

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Training Time: {total_time:.2f}s")

    return model, history, {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "training_time_seconds": total_time,
        "class_names": class_names,
    }

if __name__ == "__main__":
    my_model, my_history, my_result = my_train_feature_extraction()