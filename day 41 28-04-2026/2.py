"""Bai 2 - Phan loai cam xuc comment bang PyTorch, DUNG nn.LSTM.

Cung cau truc voi bai 1 nhung phan LSTM thay bang nn.LSTM co san.
Gom logic vao 1 class MyLSTM:
  - forward   : embed -> nn.LSTM -> lay hidden cuoi -> Linear phan loai
  - accuracy / train_step (backward) / evaluate / fit
Ben ngoai class chi con ham xu ly data va ham goi class de train.
Data: sentiment_data.csv (phan loai cam xuc comment thanh 3 lop 0/1/2).
"""

# import thu vien
import re
from collections import Counter

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Tham so gom chung mot cho de chinh
CONFIG = {
    "csv_path": "sentiment_data.csv",
    "seed": 42,            # co dinh de tai lap
    "max_samples": 200000, # lay bot cho nhe (data goc ~241k dong)
    "max_vocab": 10000,    # so tu giu lai trong tu dien
    "seq_len": 30,         # do dai chuoi (so buoc thoi gian T), cat/pad ve day
    "embed_dim": 64,       # so chieu vector nhung tu
    "hidden_size": 128,    # so chieu trang thai an cua LSTM
    "num_classes": 3,      # 3 lop cam xuc: 0, 1, 2
    "batch_size": 64,
    "lr": 0.005,
    "epochs": 10,
    "train_ratio": 0.8,
}


# ----------------------------- CLASS LSTM (dung nn.LSTM) -----------------------------
# Cung y tuong bai 1 nhung khong tu code tung gate nua, dung thang nn.LSTM.

class MyLSTM(nn.Module):

    # ---------- class chinh ----------
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        # Embedding: doi id tu -> vector; padding_idx=0 de tu <pad> luon vector 0
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # nn.LSTM co san (thay cho class con Cell tu viet o bai 1)
        # batch_first=True de input dang (batch, seq_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)     # phan loai tu hidden cuoi

    def forward(self, x):
        # x: (batch, seq_len) cac id tu
        emb = self.embed(x)                 # -> (batch, seq_len, embed_dim)
        # nn.LSTM tra ve: output moi buoc, va (h_n, c_n) trang thai cuoi
        out, (h_n, c_n) = self.lstm(emb)
        h_last = h_n[-1]                    # hidden state cuoi cung: (batch, hidden_size)
        return self.fc(h_last)             # dung hidden cuoi de phan loai

    # ---------- cac method ho tro hoc ----------
    @staticmethod
    def accuracy(logits, y_true):
        preds = logits.argmax(dim=1)
        return (preds == y_true).float().mean().item()

    def train_step(self, xb, yb, loss_fn, optimizer):
        # 1 buoc hoc: forward -> loss -> backward -> update
        self.train()
        optimizer.zero_grad()
        logits = self(xb)             # forward qua nn.LSTM
        loss = loss_fn(logits, yb)
        loss.backward()               # backward: autograd lan nguoc (BPTT)
        optimizer.step()              # cap nhat trong so
        return loss.item(), self.accuracy(logits, yb)

    @torch.no_grad()
    def evaluate(self, X, y, loss_fn, batch_size):
        # Chay tren tap test theo tung batch (khong tinh gradient)
        self.eval()
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i + batch_size], y[i:i + batch_size]
            logits = self(xb)
            total_loss += loss_fn(logits, yb).item() * len(xb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
        return total_loss / len(X), total_correct / len(X)

    def fit(self, X_train, y_train, X_test, y_test, cfg, loss_fn, optimizer):
        # Vong lap huan luyen nhieu epoch, moi epoch chia batch va xao tron
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        n = len(X_train)
        for epoch in range(1, cfg["epochs"] + 1):
            perm = torch.randperm(n, device=X_train.device)  # xao tron thu tu
            run_loss, run_correct = 0.0, 0
            for i in range(0, n, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                xb, yb = X_train[idx], y_train[idx]
                loss, _ = self.train_step(xb, yb, loss_fn, optimizer)
                run_loss += loss * len(xb)
                run_correct += int(self.accuracy(self(xb), yb) * len(xb))

            tr_loss = run_loss / n
            tr_acc = run_correct / n
            te_loss, te_acc = self.evaluate(X_test, y_test, loss_fn, cfg["batch_size"])

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
                  f"| test loss {te_loss:.4f} acc {te_acc:.3f}")
        return history


# ----------------------------- HAM XU LY DATA (ben ngoai class) -----------------------------
def tokenize(text):
    # ha chu thuong, chi giu chu va so, tach theo khoang trang
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_data(cfg):
    """Doc csv -> xay tu dien -> ma hoa cau thanh day id -> chia train/test."""
    df = pd.read_csv(cfg["csv_path"])[["Comment", "Sentiment"]].dropna()
    df = df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    df = df.iloc[:cfg["max_samples"]].reset_index(drop=True)

    # Xay tu dien tu tap cau, giu lai max_vocab tu hay gap nhat
    counter = Counter()
    for cmt in df["Comment"]:
        counter.update(tokenize(cmt))
    # id 0 = <pad>, id 1 = <unk>
    itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(cfg["max_vocab"] - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    vocab_size = len(itos)

    # Ma hoa 1 cau thanh day id, cat hoac pad ve dung seq_len
    def encode(text):
        ids = [stoi.get(tok, 1) for tok in tokenize(text)]  # tu la -> 1 (<unk>)
        ids = ids[:cfg["seq_len"]]
        ids = ids + [0] * (cfg["seq_len"] - len(ids))
        return ids

    X = torch.tensor([encode(c) for c in df["Comment"]], dtype=torch.long)
    y = torch.tensor(df["Sentiment"].values, dtype=torch.long)

    n_train = int(len(X) * cfg["train_ratio"])
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test, vocab_size


# ----------------------------- HAM GOI CLASS DE TRAIN -----------------------------
def run_training(X_train, y_train, X_test, y_test, vocab_size, cfg, device):
    # Dua data len device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Dung model tu class MyLSTM
    model = MyLSTM(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=cfg["num_classes"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Goi method fit cua class de huan luyen (co ca forward + backward ben trong)
    history = model.fit(X_train, y_train, X_test, y_test, cfg, loss_fn, optimizer)
    return model, history


def plot_history(history, cfg):
    """Ve loss va accuracy theo epoch."""
    epochs_axis = range(1, cfg["epochs"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs_axis, history["train_loss"], label="train")
    axes[0].plot(epochs_axis, history["test_loss"], label="test")
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs_axis, history["train_acc"], label="train")
    axes[1].plot(epochs_axis, history["test_acc"], label="test")
    axes[1].set_title("Accuracy theo epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ----------------------------- MAIN -----------------------------
def main():
    torch.manual_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device dang dung:", device)

    X_train, y_train, X_test, y_test, vocab_size = load_data(CONFIG)
    print("Kich thuoc tu dien:", vocab_size, "| Train:", X_train.shape, "| Test:", X_test.shape)

    model, history = run_training(
        X_train, y_train, X_test, y_test,
        vocab_size=vocab_size, cfg=CONFIG, device=device,
    )
    print("\nDo chinh xac test cuoi cung:", round(history["test_acc"][-1], 3))

    plot_history(history, CONFIG)


if __name__ == "__main__":
    main()
