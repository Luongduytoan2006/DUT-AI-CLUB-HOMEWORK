"""Bai 2 - Seq2Seq ket hop Bahdanau Attention dich Ngay Thang.

Yeu cau:
 - Kien truc Seq2Seq, dung nn.LSTM giong Bai 1.
 - Dich kieu ngay thang con nguoi doc ("12, thg 9 2010") sang chuan may tinh ("2010-09-12").
 - Xu ly o muc ky tu (char-level).
 - Tu sinh du lieu bang ham make_date_dataset, khong can load file.
 - Ve attention de xem mo hinh chu y vao chu so nao khi sinh ngay/thang/nam.
"""

import random
import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Tham so gom chung
CONFIG = {
    "seed": 42,
    "num_samples": 20000,   # So luong mau tu sinh
    "seq_len": 20,          # Do dai chuoi ky tu toi da
    "embed_dim": 32,        # So chieu vector ky tu (char embeddings)
    "hidden_size": 64,      # So chieu trang thai an LSTM
    "batch_size": 64,
    "lr": 0.005,
    "epochs": 15,
    "train_ratio": 0.8,
    "teacher_forcing": 0.5,
}

# Dinh nghia cac token dac biet
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


# ----------------------------- CAC CLASS MODEL (giong Bai 1) -----------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embed(x)
        outputs, (hidden, cell) = self.lstm(embedded)

        h_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        c_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)

        h = torch.tanh(self.fc_h(h_cat)).unsqueeze(0)
        c = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)

        return outputs, h, c


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, enc_outputs):
        hidden_expanded = hidden.unsqueeze(1)
        energy = torch.tanh(self.W1(enc_outputs) + self.W2(hidden_expanded))
        score = self.V(energy)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), enc_outputs)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, enc_outputs):
        embedded = self.embed(x)
        context, attn_weights = self.attention(hidden.squeeze(0), enc_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)
        x = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(x, hidden, cell, enc_outputs)
            outputs[:, t, :] = prediction
            top1 = prediction.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            x = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

    @torch.no_grad()
    def translate(self, src_tensor, max_len=20):
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


# ----------------------------- HAM XU LY DU LIEU DAY THANG -----------------------------

def make_date_dataset(n_samples):
    """Sinh ngau nhien n mau ngay thang nam.
    Tra ve: list ky tu cua (human_date, machine_date)."""
    start_date = datetime.date(1950, 1, 1)
    end_date = datetime.date(2050, 12, 31)
    days_between = (end_date - start_date).days
    
    months = ["thg 1", "thg 2", "thg 3", "thg 4", "thg 5", "thg 6", 
              "thg 7", "thg 8", "thg 9", "thg 10", "thg 11", "thg 12"]
              
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        random_days = random.randrange(days_between)
        dt = start_date + datetime.timedelta(days=random_days)
        
        # Ngay chuan may tinh
        machine_date = dt.strftime("%Y-%m-%d")
        
        # Ngay do con nguoi viet (VD: 12, thg 9 2010)
        human_date = f"{dt.day}, {months[dt.month - 1]} {dt.year}"
        
        inputs.append(list(human_date))
        targets.append(list(machine_date))
        
    return inputs, targets

def build_char_vocab(data):
    chars = set()
    for seq in data:
        chars.update(seq)
    
    itos = SPECIAL_TOKENS + sorted(list(chars))
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos

def encode_sequence(chars, stoi, seq_len):
    ids = [stoi.get(c, UNK_IDX) for c in chars]
    ids = ids[:seq_len-2]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    ids = ids + [PAD_IDX] * (seq_len - len(ids))
    return ids

def load_data(cfg):
    inputs, targets = make_date_dataset(cfg["num_samples"])
    
    # Vocab input va output dung chung duoc vi deu la ky tu/chu so
    vocab_stoi, vocab_itos = build_char_vocab(inputs + targets)
    
    X = torch.tensor([encode_sequence(seq, vocab_stoi, cfg["seq_len"]) for seq in inputs], dtype=torch.long)
    Y = torch.tensor([encode_sequence(seq, vocab_stoi, cfg["seq_len"]) for seq in targets], dtype=torch.long)
    
    n_train = int(len(X) * cfg["train_ratio"])
    train_data = (X[:n_train], Y[:n_train])
    test_data = (X[n_train:], Y[n_train:])
    
    return train_data, test_data, vocab_stoi, vocab_itos


# ----------------------------- HAM HUAN LUYEN VA DANH GIA -----------------------------

def run_training(train_data, test_data, vocab_size, cfg, device):
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    # Vocab input va output bang nhau
    encoder = Encoder(vocab_size, cfg["embed_dim"], cfg["hidden_size"])
    decoder = Decoder(vocab_size, cfg["embed_dim"], cfg["hidden_size"])
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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


def plot_attention(src_chars, trg_chars, attention_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, 
                xticklabels=src_chars, 
                yticklabels=trg_chars, 
                cmap="Blues", 
                annot=False)
    plt.xlabel("Ngay con nguoi doc (Source)")
    plt.ylabel("Ngay may tinh (Target)")
    plt.title("Attention Heatmap Dich Ngay Thang")
    plt.show()


# ----------------------------- MAIN -----------------------------

def main():
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    print("Dang tao du lieu ngay thang...")
    train_data, test_data, vocab_stoi, vocab_itos = load_data(CONFIG)
    vocab_size = len(vocab_itos)
    print(f"Kich thuoc tap ky tu (char-level): {vocab_size}")
    print(f"Tap train: {train_data[0].shape}, Tap test: {test_data[0].shape}")
    
    print("Dang huan luyen mo hinh...")
    model, history = run_training(
        train_data, test_data, 
        vocab_size, CONFIG, device
    )
    
    # Test thu va ve attention
    print("\nThu nghiem dich may va ve Attention:")
    test_idx = random.randint(0, len(test_data[0]) - 1)
    src_tensor = test_data[0][test_idx].unsqueeze(0).to(device)
    
    # Chuyen id thanh ky tu
    src_ids = src_tensor.squeeze(0).cpu().numpy()
    src_chars = [vocab_itos[i] for i in src_ids if i not in [PAD_IDX]]
    print("Ngay doc:", "".join(src_chars))
    
    out_ids, attentions = model.translate(src_tensor)
    out_chars = [vocab_itos[i] for i in out_ids]
    print("Ngay may:", "".join(out_chars).replace("<eos>", ""))
    
    import numpy as np
    attn_matrix = np.array(attentions)[:, :len(src_chars)]
    
    plot_attention(src_chars, out_chars, attn_matrix)

if __name__ == "__main__":
    main()
