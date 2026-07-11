"""Bai 1 - Encoder cua Transformer cho bai toan phan loai cam xuc comment.

Yeu cau: ap dung kien truc ENCODER cua Transformer de phan loai cam xuc
(3 lop: 0 tieu cuc, 1 trung tinh, 2 tich cuc). Data: sentiment_data.csv.

Luong chinh nam trong class TransformerEncoderClassifier, xep theo thu tu bai giang:
  Embedding -> PositionalEncoding -> [EncoderLayer] x N -> pooling -> Linear phan loai.

Cac khoi cua transformer duoc tach thanh nhieu class rieng (PositionalEncoding,
MultiHeadSelfAttention, PositionwiseFeedForward, EncoderLayer) roi ghep lai
trong class chinh. Comment tieng Viet khong dau.
"""

# import thu vien
import re
import math
from collections import Counter

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Tham so gom chung mot cho de chinh
CONFIG = {
    "csv_path": "sentiment_data.csv",
    "seed": 42,           # co dinh de tai lap
    "max_samples": 200000, # lay bot cho nhe (data goc ~241k dong)
    "max_vocab": 15000,   # so tu giu lai trong tu dien
    "seq_len": 40,        # do dai chuoi, cat/pad ve day
    "embed_dim": 128,     # so chieu vector nhung tu (chia het cho num_heads)
    "num_heads": 4,       # so dau self-attention
    "ff_dim": 256,        # so chieu lop feed forward giua
    "num_layers": 2,      # so lop EncoderLayer xep chong
    "dropout": 0.1,       # ty le dropout chong overfit
    "num_classes": 3,     # 3 lop cam xuc: 0, 1, 2
    "batch_size": 128,
    "lr": 0.0005,
    "epochs": 8,
    "train_ratio": 0.8,
}

PAD_IDX = 0   # id cua token <pad>
UNK_IDX = 1   # id cua token <unk>


# ----------------------------- CAC LOP CUA TRANSFORMER -----------------------------
class PositionalEncoding(nn.Module):
    """Them vector vi tri (sinusoidal) vao embedding.

    Self-attention khong biet thu tu tu, nen phai cong them thong tin vi tri.
    Cong thuc goc trong paper "Attention is all you need":
      PE(pos, 2i)   = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        # pe: (max_len, embed_dim) tinh truoc, khong hoc
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()      # (max_len, 1)
        # div_term: 1 / 10000^(2i/d) tinh o dang log cho on dinh
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div_term)   # cot chan dung sin
        pe[:, 1::2] = torch.cos(pos * div_term)   # cot le dung cos
        # register_buffer: luu cung model nhung khong phai tham so hoc
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim) -> cong them PE cua dung seq_len dau
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Self-attention nhieu dau (multi-head).

    Chia embedding thanh num_heads dau, moi dau tu tinh attention rieng:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    roi ghep lai va chieu ve embed_dim.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phai chia het cho num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads    # so chieu moi dau

        # 3 phep chieu tuyen tinh tao Q, K, V va 1 phep chieu ghep dau ra
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.shape

        # Chieu tao Q, K, V roi tach thanh nhieu dau:
        # (batch, seq_len, embed) -> (batch, num_heads, seq_len, head_dim)
        def split_heads(t):
            t = t.view(batch, seq_len, self.num_heads, self.head_dim)
            return t.transpose(1, 2)

        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        # Diem tuong dong Q.K^T, chia sqrt(head_dim) cho on dinh gradient
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask: cho gia tri padding -> so am rat lon de softmax lam trong so ~ 0
        # (dung -1e9 thay -inf de tranh NaN neu ca hang deu bi mask)
        if mask is not None:
            # mask: (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)   # trong so chu y (tong = 1)
        context = torch.matmul(attn, V)        # (batch, num_heads, seq_len, head_dim)

        # Ghep cac dau lai: -> (batch, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.W_o(context)               # chieu ghep dau ra


class PositionwiseFeedForward(nn.Module):
    """Mang feed forward ap dung doc lap cho tung vi tri (2 lop Linear + ReLU)."""
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)   # mo rong chieu
        self.fc2 = nn.Linear(ff_dim, embed_dim)   # nen ve lai
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Mot lop Encoder = Self-Attention + Feed Forward, moi khoi co residual + LayerNorm.

    So do (kieu post-norm):
      x -> +Attention(x)     -> LayerNorm
        -> +FeedForward(...)  -> LayerNorm
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Khoi 1: self-attention + residual (cong lai dau vao) + chuan hoa
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Khoi 2: feed forward + residual + chuan hoa
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ----------------------------- CLASS MO HINH CHINH -----------------------------
# Day la luong chinh cua Transformer Encoder, ghep cac class layer o tren lai.

class TransformerEncoderClassifier(nn.Module):
    """Encoder Transformer + dau phan loai cam xuc.

    Trinh tu forward:
      1. Embedding    : id tu -> vector, nhan sqrt(embed_dim) theo paper
      2. PositionalEncoding : cong them thong tin vi tri
      3. N x EncoderLayer   : self-attention + feed forward (co mask padding)
      4. Mean pooling       : gop cac token (bo qua padding) thanh 1 vector cau
      5. Linear             : chieu ve so lop de phan loai
    """
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        # Embedding: doi id tu -> vector; padding_idx=PAD_IDX de <pad> luon vector 0
        self.embed = nn.Embedding(vocab_size, cfg["embed_dim"], padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(cfg["embed_dim"], max_len=cfg["seq_len"])
        self.dropout = nn.Dropout(cfg["dropout"])

        # Xep chong num_layers lop EncoderLayer giong nhau
        self.layers = nn.ModuleList([
            EncoderLayer(cfg["embed_dim"], cfg["num_heads"], cfg["ff_dim"], cfg["dropout"])
            for _ in range(cfg["num_layers"])
        ])

        self.fc = nn.Linear(cfg["embed_dim"], cfg["num_classes"])  # dau phan loai

    @staticmethod
    def make_mask(x):
        # Tao mask danh dau vi tri KHONG phai padding: (batch, 1, 1, seq_len)
        return (x != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        # x: (batch, seq_len) cac id tu
        mask = self.make_mask(x)

        # 1 + 2: embedding (co scale) roi cong positional encoding
        emb = self.embed(x) * (self.embed_dim ** 0.5)
        emb = self.dropout(self.pos_enc(emb))

        # 3: cho qua tung lop encoder
        out = emb
        for layer in self.layers:
            out = layer(out, mask)

        # 4: mean pooling bo qua padding (dung mask lam trong so)
        mask_f = (x != PAD_IDX).unsqueeze(-1).float()   # (batch, seq_len, 1)
        summed = (out * mask_f).sum(dim=1)              # tong cac token that
        count = mask_f.sum(dim=1).clamp(min=1e-9)       # so token that (tranh chia 0)
        pooled = summed / count                         # (batch, embed_dim)

        # 5: phan loai
        return self.fc(pooled)

    # ---------- cac method ho tro hoc ----------
    @staticmethod
    def accuracy(logits, y_true):
        preds = logits.argmax(dim=1)
        return (preds == y_true).float().mean().item()

    def train_step(self, xb, yb, loss_fn, optimizer):
        # 1 buoc hoc: forward -> loss -> backward -> update
        self.train()
        optimizer.zero_grad()
        logits = self(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
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
            self.train()
            perm = torch.randperm(n, device=X_train.device)   # xao tron thu tu
            run_loss, run_correct = 0.0, 0
            for i in range(0, n, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                xb, yb = X_train[idx], y_train[idx]
                loss, acc = self.train_step(xb, yb, loss_fn, optimizer)
                run_loss += loss * len(xb)
                run_correct += int(acc * len(xb))

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
    # Loai cau rong sau khi tokenize (vd comment toan emoji) de tranh cau toan padding
    df = df[df["Comment"].apply(lambda c: len(tokenize(c)) > 0)]
    df = df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    df = df.iloc[:cfg["max_samples"]].reset_index(drop=True)

    # Xay tu dien tu tap cau, giu lai max_vocab tu hay gap nhat
    counter = Counter()
    for cmt in df["Comment"]:
        counter.update(tokenize(cmt))
    itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(cfg["max_vocab"] - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    vocab_size = len(itos)

    # Ma hoa 1 cau thanh day id, cat hoac pad ve dung seq_len
    def encode(text):
        ids = [stoi.get(tok, UNK_IDX) for tok in tokenize(text)]
        ids = ids[:cfg["seq_len"]]
        ids = ids + [PAD_IDX] * (cfg["seq_len"] - len(ids))
        return ids

    X = torch.tensor([encode(c) for c in df["Comment"]], dtype=torch.long)
    y = torch.tensor(df["Sentiment"].values, dtype=torch.long)

    n_train = int(len(X) * cfg["train_ratio"])
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test, vocab_size, itos


# ----------------------------- HAM GOI CLASS DE TRAIN -----------------------------
def run_training(X_train, y_train, X_test, y_test, vocab_size, cfg, device):
    # Dua data len device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Dung model tu class chinh
    model = TransformerEncoderClassifier(vocab_size, cfg).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Goi method fit cua class de huan luyen (forward + backward ben trong)
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

    X_train, y_train, X_test, y_test, vocab_size, itos = load_data(CONFIG)
    print("Kich thuoc tu dien:", vocab_size, "| Train:", X_train.shape, "| Test:", X_test.shape)

    model, history = run_training(
        X_train, y_train, X_test, y_test,
        vocab_size=vocab_size, cfg=CONFIG, device=device,
    )
    print("\nDo chinh xac test cuoi cung:", round(history["test_acc"][-1], 3))

    plot_history(history, CONFIG)


if __name__ == "__main__":
    main()
