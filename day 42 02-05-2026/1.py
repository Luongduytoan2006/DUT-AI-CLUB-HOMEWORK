import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter
import sys
import os

# Set encoding for Windows terminal to handle Vietnamese characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- 1. Tiền xử lý dữ liệu (Data Preprocessing) ---

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word.lower())

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def tokenize_and_build_vocab(en_sents, vi_sents):
    en_vocab = Vocabulary("English")
    vi_vocab = Vocabulary("Vietnamese")
    
    for en, vi in zip(en_sents, vi_sents):
        en_vocab.add_sentence(en)
        vi_vocab.add_sentence(vi)
        
    return en_vocab, vi_vocab

def sentence_to_tensor(vocab, sentence, device):
    indexes = [vocab.word2index.get(word.lower(), vocab.word2index["<unk>"]) for word in sentence.split()]
    indexes.append(vocab.word2index["<eos>"])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# --- 2. Kiến trúc Mô hình (Model Architecture) ---

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden: [n_layers, batch_size, hid_dim]
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        input = input.unsqueeze(0) # [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

class TranslatorModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

# --- 3. Các thuật toán Decoding ---

def decode_greedy(model, src_tensor, max_len, vi_vocab):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
    
    trg_indexes = [vi_vocab.word2index["<sos>"]]
    
    for i in range(max_len):
        trg_input = torch.tensor([trg_indexes[-1]], device=model.device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_input, hidden)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == vi_vocab.word2index["<eos>"]:
            break
            
    return [vi_vocab.index2word[idx] for idx in trg_indexes]

def decode_sampling(model, src_tensor, max_len, vi_vocab, temperature=1.0):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
    
    trg_indexes = [vi_vocab.word2index["<sos>"]]
    
    for i in range(max_len):
        trg_input = torch.tensor([trg_indexes[-1]], device=model.device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_input, hidden)
        
        probs = F.softmax(output / temperature, dim=1)
        pred_token = torch.multinomial(probs, 1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == vi_vocab.word2index["<eos>"]:
            break
            
    return [vi_vocab.index2word[idx] for idx in trg_indexes]

def decode_beam_search(model, src_tensor, max_len, vi_vocab, beam_size=3):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
    
    # beam: (score, [indices], hidden_state)
    beams = [(0, [vi_vocab.word2index["<sos>"]], hidden)]
    
    for i in range(max_len):
        candidates = []
        for score, indices, h in beams:
            if indices[-1] == vi_vocab.word2index["<eos>"]:
                candidates.append((score, indices, h))
                continue
                
            trg_input = torch.tensor([indices[-1]], device=model.device)
            with torch.no_grad():
                output, new_h = model.decoder(trg_input, h)
            
            log_probs = F.log_softmax(output, dim=1)
            topk_probs, topk_idx = log_probs.topk(beam_size)
            
            for j in range(beam_size):
                candidates.append((score + topk_probs[0][j].item(), indices + [topk_idx[0][j].item()], new_h))
        
        # Sắp xếp và lấy top beam_size
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # Nếu tất cả beams đều kết thúc bằng <eos>
        if all(b[1][-1] == vi_vocab.word2index["<eos>"] for b in beams):
            break
            
    best_indices = beams[0][1]
    return [vi_vocab.index2word[idx] for idx in best_indices]

# --- 4. Đánh giá và So sánh ---

def evaluate_strategies(model, en_sentence, en_vocab, vi_vocab, device):
    src_tensor = sentence_to_tensor(en_vocab, en_sentence, device)
    
    greedy_res = decode_greedy(model, src_tensor, 20, vi_vocab)
    sampling_res = decode_sampling(model, src_tensor, 20, vi_vocab, temperature=0.7)
    beam_res = decode_beam_search(model, src_tensor, 20, vi_vocab, beam_size=3)
    
    print(f"\nSource Sentence: {en_sentence}")
    print(f"Greedy Strategy: {' '.join(greedy_res)}")
    print(f"Sampling Strategy: {' '.join(sampling_res)}")
    print(f"Beam Search Strategy: {' '.join(beam_res)}")

# --- 5. Huấn luyện (Training Logic) ---

def train_one_epoch(model, training_pairs, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    
    for src, trg in training_pairs:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        encoder_hidden = model.encoder(src)
        
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = model.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        hidden = encoder_hidden
        input = trg[0, :] # <sos>
        
        for t in range(1, trg_len):
            output, hidden = model.decoder(input, hidden)
            outputs[t] = output
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if use_teacher_forcing else top1
            
        outputs = outputs[1:].view(-1, trg_vocab_size)
        trg = trg[1:].view(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(training_pairs)

# --- 6. Chạy chính (Main Function) ---

def run_main():
    # Cấu hình
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = 0 # Sẽ cập nhật sau khi build vocab
    OUTPUT_DIM = 0
    EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 1 # Để đơn giản cho demo
    N_EPOCHS = 20
    
    # Dữ liệu mẫu (Thay thế bằng load file từ Kaggle)
    en_sents = ["hello", "how are you", "good morning", "i am fine", "thank you"]
    vi_sents = ["xin chào", "bạn khỏe không", "chào buổi sáng", "tôi khỏe", "cảm ơn bạn"]
    
    # Thử load file nếu tồn tại
    if os.path.exists("en_sents.txt") and os.path.exists("vi_sents.txt"):
        with open("en_sents.txt", "r", encoding="utf-8") as f: en_sents = f.read().splitlines()[:100] # Lấy ít để demo
        with open("vi_sents.txt", "r", encoding="utf-8") as f: vi_sents = f.read().splitlines()[:100]
        print("Loaded dataset from files.")
    else:
        print("Using dummy dataset for demonstration.")

    en_vocab, vi_vocab = tokenize_and_build_vocab(en_sents, vi_sents)
    
    # Chuẩn bị training data
    training_pairs = []
    for en, vi in zip(en_sents, vi_sents):
        src = sentence_to_tensor(en_vocab, en, DEVICE)
        trg = sentence_to_tensor(vi_vocab, vi, DEVICE)
        training_pairs.append((src, trg))
    
    # Khởi tạo mô hình
    enc = EncoderRNN(en_vocab.n_words, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = DecoderRNN(vi_vocab.n_words, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = TranslatorModel(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vi_vocab.word2index["<pad>"])
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(N_EPOCHS):
        loss = train_one_epoch(model, training_pairs, optimizer, criterion, DEVICE)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch: {epoch+1:02} | Loss: {loss:.4f}")
            
    # So sánh các chiến lược
    test_sentence = "how are you"
    evaluate_strategies(model, test_sentence, en_vocab, vi_vocab, DEVICE)

if __name__ == "__main__":
    run_main()
