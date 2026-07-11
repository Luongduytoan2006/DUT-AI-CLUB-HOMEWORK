"""Bai 2 - Mo hinh Seq2Seq Transformer cho dich may Anh-Viet.

Yeu cau:
- Su dung kien truc Transformer de dich ngon ngu.
- So sanh 3 chien luoc sinh tu (Decoding Strategies): Greedy Search, Beam Search, Sampling.
"""

import math
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------- CONFIG & SPECIAL TOKENS -----------------------------
CONFIG = {
    "en_path": "../day 43 16-05-2026/data/en_sents",
    "vi_path": "../day 43 16-05-2026/data/vi_sents",
    "seed": 42,
    "max_samples": 20000,   # So luong mau data de huan luyen
    "max_vocab": 8000,
    "seq_len": 20,
    "embed_dim": 256,
    "num_heads": 8,
    "ff_dim": 512,
    "num_layers": 3,
    "dropout": 0.1,
    "batch_size": 64,
    "lr": 0.0005,
    "epochs": 10,
    "train_ratio": 0.9,
}

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


# ----------------------------- CAC LOP CO BAN -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# ----------------------------- ENCODER & DECODER -----------------------------

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=trg_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, mask=src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    """Mo hinh Transformer ket hop Encoder va Decoder tu code."""
    def __init__(self, src_vocab, trg_vocab, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.src_embed = nn.Embedding(src_vocab, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(trg_vocab, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(embed_dim)
        
        self.encoder = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(embed_dim, trg_vocab)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        pad_mask = (trg != PAD_IDX).unsqueeze(1).unsqueeze(2)
        seq_len = trg.shape[1]
        sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=trg.device)).bool()
        return pad_mask & sub_mask

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.src_embed(src) * math.sqrt(self.embed_dim)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, trg, enc_out, trg_mask, src_mask):
        x = self.dropout(self.pos_enc(self.trg_embed(trg) * math.sqrt(self.embed_dim)))
        for layer in self.decoder:
            x = layer(x, enc_out, trg_mask, src_mask)
        return self.fc_out(x)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encode(src, src_mask)
        return self.decode(trg, enc_out, trg_mask, src_mask)


# ----------------------------- DATA PIPELINE -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", " ", text)
    return text.strip().split()

def build_vocab(sentences, max_vocab):
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)
    itos = SPECIAL_TOKENS + [w for w, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS))]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode_sentence(tokens, stoi, seq_len):
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens][:seq_len-2]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    ids += [PAD_IDX] * (seq_len - len(ids))
    return ids

def load_data(cfg):
    with open(cfg["en_path"], "r", encoding="utf-8") as f:
        en_lines = f.readlines()
    with open(cfg["vi_path"], "r", encoding="utf-8") as f:
        vi_lines = f.readlines()
        
    data = list(zip(en_lines, vi_lines))
    random.seed(cfg["seed"])
    random.shuffle(data)
    data = data[:cfg["max_samples"]]
    
    en_toks = [clean_text(en) for en, vi in data]
    vi_toks = [clean_text(vi) for en, vi in data]
    
    en_stoi, en_itos = build_vocab(en_toks, cfg["max_vocab"])
    vi_stoi, vi_itos = build_vocab(vi_toks, cfg["max_vocab"])
    
    X = torch.tensor([encode_sentence(toks, en_stoi, cfg["seq_len"]) for toks in en_toks], dtype=torch.long)
    Y = torch.tensor([encode_sentence(toks, vi_stoi, cfg["seq_len"]) for toks in vi_toks], dtype=torch.long)
    
    n_train = int(len(X) * cfg["train_ratio"])
    train_data = (X[:n_train], Y[:n_train])
    test_data = (X[n_train:], Y[n_train:])
    
    return train_data, test_data, en_stoi, en_itos, vi_stoi, vi_itos


# ----------------------------- TRAINING -----------------------------

def run_training(train_data, en_vocab_size, vi_vocab_size, cfg, device):
    X_train, Y_train = train_data
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    model = Seq2SeqTransformer(
        en_vocab_size, vi_vocab_size, cfg["embed_dim"], 
        cfg["num_layers"], cfg["num_heads"], cfg["ff_dim"], cfg["dropout"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    n = len(X_train)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(n, device=device)
        
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            src, trg = X_train[idx], Y_train[idx]
            
            optimizer.zero_grad()
            # Dua trg vao decoder (bo ky tu cuoi de shift sang phai)
            output = model(src, trg[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_target = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(idx)
            
        print(f"Epoch {epoch:2d} | Train Loss: {epoch_loss/n:.4f}")
        
    return model


# ----------------------------- DECODING STRATEGIES -----------------------------

def greedy_search(model, src_tensor, max_len, device):
    """1. Greedy Search: Chon tu co xac suat cao nhat tai moi buoc."""
    model.eval()
    src_mask = model.make_src_mask(src_tensor)
    enc_out = model.encode(src_tensor, src_mask)
    
    trg_indexes = [SOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes], device=device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            out = model.decode(trg_tensor, enc_out, trg_mask, src_mask)
            
        pred_token = out[:, -1, :].argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
            
    return trg_indexes


def beam_search(model, src_tensor, max_len, device, beam_width=3):
    """2. Beam Search: Giu lai top-K (beam_width) chuoi tot nhat tai moi buoc de phong ngua sai lam cuc bo."""
    model.eval()
    src_mask = model.make_src_mask(src_tensor)
    enc_out = model.encode(src_tensor, src_mask)
    
    beams = [([SOS_IDX], 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == EOS_IDX:
                all_candidates.append((seq, score))
                continue
                
            trg_tensor = torch.tensor([seq], device=device)
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                out = model.decode(trg_tensor, enc_out, trg_mask, src_mask)
                
            # Log xac suat cua tu cuoi cung de cong vao score
            log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                next_seq = seq + [topk_idx[i].item()]
                next_score = score + topk_log_probs[i].item()
                all_candidates.append((next_seq, next_score))
                
        # Giu lai top-K chuoi co diem cao nhat hien tai
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        
        # Dung lai neu tat ca cac beam tot nhat deu da sinh ra EOS
        if all(b[0][-1] == EOS_IDX for b in beams):
            break
            
    return beams[0][0] # Tra ve chuoi co diem cao nhat (top 1)


def sampling_search(model, src_tensor, max_len, device, temperature=1.0):
    """3. Sampling: Chon tu tiep theo mang tinh ngau nhien dua tren xac suat, tao su da dang (co ban chat sang tao)."""
    model.eval()
    src_mask = model.make_src_mask(src_tensor)
    enc_out = model.encode(src_tensor, src_mask)
    
    trg_indexes = [SOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes], device=device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            out = model.decode(trg_tensor, enc_out, trg_mask, src_mask)
            
        # Phat trien nhiet do (temperature)
        logits = out[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        
        # Lay ngau nhien dua tren phan bo xac suat
        next_token = torch.multinomial(probs, num_samples=1).item()
        trg_indexes.append(next_token)
        
        if next_token == EOS_IDX:
            break
            
    return trg_indexes


# ----------------------------- MAIN DE KIEM TRA -----------------------------
def main():
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    print("Dang xu ly du lieu...")
    train_data, test_data, en_stoi, en_itos, vi_stoi, vi_itos = load_data(CONFIG)
    
    print("Huan luyen mo hinh Transformer...")
    model = run_training(train_data, len(en_itos), len(vi_itos), CONFIG, device)
    
    print("\n--- TEST DECODING STRATEGIES ---")
    test_idx = random.randint(0, len(test_data[0])-1)
    src_tensor = test_data[0][test_idx].unsqueeze(0).to(device)
    
    src_tokens = [en_itos[i] for i in src_tensor.squeeze(0).cpu().numpy() if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
    print("Cau goc (EN):", " ".join(src_tokens))
    
    # Ham ho tro chuyen indexes -> chuoi
    def ids_to_string(ids):
        return " ".join([vi_itos[i] for i in ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)])

    # 1. Greedy
    greedy_ids = greedy_search(model, src_tensor, CONFIG["seq_len"], device)
    print("1. Greedy Search:", ids_to_string(greedy_ids))
    
    # 2. Beam Search
    beam_ids = beam_search(model, src_tensor, CONFIG["seq_len"], device, beam_width=3)
    print("2. Beam Search  :", ids_to_string(beam_ids))
    
    # 3. Sampling
    sample_ids = sampling_search(model, src_tensor, CONFIG["seq_len"], device, temperature=0.8)
    print("3. Sampling     :", ids_to_string(sample_ids))

if __name__ == "__main__":
    main()
