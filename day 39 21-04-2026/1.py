import torch
import torch.nn as nn
import math
import re
import csv
import random
from pathlib import Path
from torch.utils.data import Dataset
from underthesea import word_tokenize

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
MAX_VOCAB = 30000
MAX_LEN = 180

class RnnCell_Manual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        k = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.W_ih, -k, k)
        nn.init.uniform_(self.W_hh, -k, k)
        nn.init.uniform_(self.b_ih, -k, k)
        nn.init.uniform_(self.b_hh, -k, k)

    def forward(self, x_t, h_prev):
        return torch.tanh(x_t @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh)

class RnnStack_Manual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                in_sz = input_size
            else:
                in_sz = hidden_size
            self.cells.append(RnnCell_Manual(in_sz, hidden_size))

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        h_prev = []
        for layer in range(self.num_layers):
            h_prev.append(h_0[layer])
            
        outputs = []
        for t in range(seq_len):
            inp = x[t]
            new_h = []
            for layer in range(len(self.cells)):
                cell = self.cells[layer]
                h = cell(inp, h_prev[layer])
                new_h.append(h)
                inp = h
            h_prev = new_h
            outputs.append(inp)
        output = torch.stack(outputs, dim=0)
        h_n = torch.stack(h_prev, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n

def load_split(split_dir):
    texts, labels = [], []
    label_map = {'neg': 0, 'pos': 1}
    for name in ['neg', 'pos']:
        class_dir = Path(split_dir) / name
        if class_dir.exists():
            for fp in class_dir.glob('*.txt'):
                texts.append(fp.read_text(encoding='utf-8', errors='ignore'))
                labels.append(label_map[name])
    return texts, labels

def clean_tokenize(text):
    text = text.strip().lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens_str = word_tokenize(text, format='text')
    tokens = tokens_str.split()
    return tokens

def encode(text, word2id, max_len=MAX_LEN):
    raw_tokens = clean_tokenize(text)
    toks = []
    for i in range(min(len(raw_tokens), max_len)):
        toks.append(raw_tokens[i])
    
    ids = []
    for t in toks:
        if t in word2id:
            ids.append(word2id[t])
        else:
            ids.append(word2id[UNK_TOKEN])
            
    length = len(ids)
    while len(ids) < max_len:
        ids.append(word2id[PAD_TOKEN])
    return ids, max(length, 1)

class TextDataset_Manual(Dataset):
    def __init__(self, texts, labels, word2id):
        self.ids = []
        self.lens = []
        self.labels = labels
        for t in texts:
            i, l = encode(t, word2id)
            self.ids.append(i)
            self.lens.append(l)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.ids[idx], dtype=torch.long),
            'length': torch.tensor(self.lens[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }

def build_vocab(train_texts, vocab_path):
    freq = {}
    for t in train_texts:
        tokens = clean_tokenize(t)
        for tok in tokens:
            if tok in freq:
                freq[tok] += 1
            else:
                freq[tok] = 1
                
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    most_common = []
    for i in range(min(len(sorted_freq), MAX_VOCAB)):
        most_common.append(sorted_freq[i])
        
    with open(vocab_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'freq'])
        writer.writerow([PAD_TOKEN, 0])
        writer.writerow([UNK_TOKEN, 0])
        for item in most_common:
            writer.writerow([item[0], item[1]])
            
    vocab_tokens = [PAD_TOKEN, UNK_TOKEN]
    for item in most_common:
        vocab_tokens.append(item[0])
        
    word2id = {}
    for i in range(len(vocab_tokens)):
        word2id[vocab_tokens[i]] = i
        
    return word2id

if __name__ == '__main__':
    rnn = RnnStack_Manual(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    dummy = torch.randn(4, 5, 10)
    out, h_n = rnn(dummy)
    print(f'Output shape : {out.shape}')
    print(f'Hidden shape : {h_n.shape}')
