"""Bai 1 - Seq2Seq ket hop Bahdanau Attention dich Viet -> Anh.

Yeu cau:
 - Kien truc Seq2Seq, dung nn.LSTM.
 - Tach biet cac class: Encoder, BahdanauAttention, Decoder, Seq2Seq.
 - Ve attention heatmap.
 - Doc data tu data/vi_sents (nguon) va data/en_sents (dich).
 - Comment tieng Viet khong dau, code de doc.
"""

import re
import random
from collections import Counter

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tham so gom chung
CONFIG = {
    "vi_path": "data/vi_sents",
    "en_path": "data/en_sents",
    "seed": 42,
    "max_samples": 30000,   # Dung mot phan data cho nhanh
    "max_vocab": 8000,      # So tu toi da trong tu dien
    "seq_len": 20,          # Do dai cau toi da
    "embed_dim": 128,       # So chieu vector tu
    "hidden_size": 256,     # So chieu trang thai an LSTM
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 15,
    "train_ratio": 0.9,
    "teacher_forcing": 0.5, # Ty le dung tu that cua tap dich lam dau vao tiep theo
}

# Dinh nghia cac token dac biet
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


# ----------------------------- CAC CLASS MODEL -----------------------------

class Encoder(nn.Module):
    """Doc cau tieng Viet, ma hoa thanh cac dac trung."""
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Dung Bidirectional LSTM de hieu ngu canh 2 chieu
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        # Giam chieu hidden state tu 2 chieu ve 1 chieu cho Decoder
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embed(x)
        # outputs: (batch, seq, hidden_size * 2)
        # hidden, cell: (2, batch, hidden_size)
        outputs, (hidden, cell) = self.lstm(embedded)

        # Ghep trang thai an cua chieu xuoi va chieu nguoc
        h_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # (batch, hidden_size * 2)
        c_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)     # (batch, hidden_size * 2)

        # Bien doi ve kich thuoc hidden_size
        h = torch.tanh(self.fc_h(h_cat)).unsqueeze(0) # (1, batch, hidden_size)
        c = torch.tanh(self.fc_c(c_cat)).unsqueeze(0) # (1, batch, hidden_size)

        return outputs, h, c


class BahdanauAttention(nn.Module):
    """Tinh toan muc do chu y cua tung tu trong cau nguon khi sinh tu dich."""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size * 2, hidden_size) # Danh cho encoder outputs (BiLSTM nen *2)
        self.W2 = nn.Linear(hidden_size, hidden_size)     # Danh cho decoder hidden state
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, enc_outputs):
        # hidden: (batch, hidden_size) (da bo chieu layer)
        # enc_outputs: (batch, seq_len, hidden_size * 2)
        hidden_expanded = hidden.unsqueeze(1) # (batch, 1, hidden_size)

        # Tinh toan nang luong / diem so cho tung tu
        energy = torch.tanh(self.W1(enc_outputs) + self.W2(hidden_expanded))
        score = self.V(energy) # (batch, seq_len, 1)

        # Dung softmax de lay trong so (tong = 1)
        attn_weights = torch.softmax(score, dim=1) # (batch, seq_len, 1)

        # Nhan trong so voi dau ra encoder de lay context vector
        context = torch.bmm(attn_weights.transpose(1, 2), enc_outputs) # (batch, 1, hidden_size * 2)

        return context, attn_weights


class Decoder(nn.Module):
    """Sinh tung tu tieng Anh dua tren context (su chu y) va tu truoc do."""
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_size)
        # Dau vao LSTM = embedding cua tu truoc do + context vector tu attention
        self.lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, enc_outputs):
        # x: (batch, 1) la id cua 1 tu tai buoc t
        embedded = self.embed(x) # (batch, 1, embed_dim)

        # Tinh context vector thong qua attention
        context, attn_weights = self.attention(hidden.squeeze(0), enc_outputs)

        # Ghep embedding va context
        lstm_input = torch.cat((embedded, context), dim=2) # (batch, 1, embed_dim + hidden*2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1)) # (batch, vocab_size)

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Mo hinh tong chua Encoder va Decoder, quan ly qua trinh train / inference."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        # Noi luu ket qua
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Buoc 1: Encoder xu ly cau nguon
        enc_outputs, hidden, cell = self.encoder(src)

        # Buoc 2: Decoder sinh tung tu, bat dau voi <sos>
        x = trg[:, 0].unsqueeze(1) # (batch, 1)

        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(x, hidden, cell, enc_outputs)
            outputs[:, t, :] = prediction

            top1 = prediction.argmax(1) # Lay tu co xac suat cao nhat
            # Teacher forcing: doi khi dung tu that thay vi tu du doan
            teacher_force = random.random() < teacher_forcing_ratio
            x = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

    @torch.no_grad()
    def translate(self, src_tensor, max_len=30):
        """Dich 1 cau va luu lai attention de ve bieu do."""
        self.eval()
        enc_outputs, hidden, cell = self.encoder(src_tensor)
        
        x = torch.tensor([[SOS_IDX]]).to(self.device)
        
        outputs = []
        attentions = []

        for _ in range(max_len):
            prediction, hidden, cell, attn_weights = self.decoder(x, hidden, cell, enc_outputs)
            attentions.append(attn_weights.squeeze(0).squeeze(1).cpu().numpy())
            
            top1 = prediction.argmax(1).item()
            outputs.append(top1)
            
            if top1 == EOS_IDX:
                break
                
            x = torch.tensor([[top1]]).to(self.device)

        return outputs, attentions


# ----------------------------- HAM XU LY DU LIEU -----------------------------

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
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens]
    ids = ids[:seq_len-2] # Chua cho cho <sos> va <eos>
    ids = [SOS_IDX] + ids + [EOS_IDX]
    ids = ids + [PAD_IDX] * (seq_len - len(ids))
    return ids

def load_data(cfg):
    """Doc 2 file song ngu, xay dung tu dien va ma hoa."""
    with open(cfg["vi_path"], "r", encoding="utf-8") as f:
        vi_lines = f.readlines()
    with open(cfg["en_path"], "r", encoding="utf-8") as f:
        en_lines = f.readlines()
        
    data = list(zip(vi_lines, en_lines))
    random.seed(cfg["seed"])
    random.shuffle(data)
    data = data[:cfg["max_samples"]]
    
    vi_toks = [clean_text(vi) for vi, en in data]
    en_toks = [clean_text(en) for vi, en in data]
    
    vi_stoi, vi_itos = build_vocab(vi_toks, cfg["max_vocab"])
    en_stoi, en_itos = build_vocab(en_toks, cfg["max_vocab"])
    
    X = torch.tensor([encode_sentence(toks, vi_stoi, cfg["seq_len"]) for toks in vi_toks], dtype=torch.long)
    Y = torch.tensor([encode_sentence(toks, en_stoi, cfg["seq_len"]) for toks in en_toks], dtype=torch.long)
    
    n_train = int(len(X) * cfg["train_ratio"])
    train_data = (X[:n_train], Y[:n_train])
    test_data = (X[n_train:], Y[n_train:])
    
    return train_data, test_data, vi_stoi, vi_itos, en_stoi, en_itos


# ----------------------------- HAM HUAN LUYEN VA DANH GIA -----------------------------

def run_training(train_data, test_data, vi_vocab_size, en_vocab_size, cfg, device):
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    encoder = Encoder(vi_vocab_size, cfg["embed_dim"], cfg["hidden_size"])
    decoder = Decoder(en_vocab_size, cfg["embed_dim"], cfg["hidden_size"])
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # Bo qua tinh loss cho ky tu padding
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    history = {"train_loss": []}
    n = len(X_train)
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(n, device=device)
        
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            src, trg = X_train[idx], Y_train[idx]
            
            optimizer.zero_grad()
            output = model(src, trg, cfg["teacher_forcing"])
            
            # output: (batch, seq, vocab), trg: (batch, seq)
            # cat bo tu <sos> o vi tri dau tien de tinh loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(idx)
            
        avg_loss = epoch_loss / n
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f}")
        
    return model, history


def plot_attention(sentence_tokens, translation_tokens, attention_matrix):
    """Ve heatmap attention the hien su tuong quan giua cau nguon va cau dich."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, 
                xticklabels=sentence_tokens, 
                yticklabels=translation_tokens, 
                cmap="viridis", 
                annot=False)
    plt.xlabel("Cau tieng Viet (Source)")
    plt.ylabel("Cau tieng Anh (Target)")
    plt.title("Bahdanau Attention Heatmap")
    plt.show()


# ----------------------------- MAIN -----------------------------

def main():
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    print("Dang doc va xu ly du lieu...")
    train_data, test_data, vi_stoi, vi_itos, en_stoi, en_itos = load_data(CONFIG)
    print(f"Vocab size - VN: {len(vi_itos)}, EN: {len(en_itos)}")
    print(f"Tap train: {train_data[0].shape}, Tap test: {test_data[0].shape}")
    
    print("Dang huan luyen mo hinh...")
    model, history = run_training(
        train_data, test_data, 
        len(vi_itos), len(en_itos), 
        CONFIG, device
    )
    
    # Test thu dich va ve attention
    print("\nThu nghiem dich may va ve Attention:")
    test_idx = 0
    src_tensor = test_data[0][test_idx].unsqueeze(0).to(device)
    
    # Lay cac token nguon bang cach loai bo padding
    src_ids = src_tensor.squeeze(0).cpu().numpy()
    src_tokens = [vi_itos[i] for i in src_ids if i not in [PAD_IDX]]
    print("Cau Viet goc:", " ".join(src_tokens))
    
    # Dich
    out_ids, attentions = model.translate(src_tensor)
    out_tokens = [en_itos[i] for i in out_ids]
    print("Cau Anh dich:", " ".join(out_tokens))
    
    # Chuan bi matrix cho heatmap
    # attentions la list cac mang numpy kich thuoc (src_len)
    # Ta chi lay chieu dai that cua cau nguon
    import numpy as np
    attn_matrix = np.array(attentions)[:, :len(src_tokens)]
    
    plot_attention(src_tokens, out_tokens, attn_matrix)

if __name__ == "__main__":
    main()
