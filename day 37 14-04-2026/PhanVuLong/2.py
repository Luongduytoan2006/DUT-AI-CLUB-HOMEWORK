#!/usr/bin/env python
# coding: utf-8

# # Bai 2 - Skip-gram (NumPy + underthesea)
# 
# Yeu cau da dap ung:
# - Tien xu ly bang underthesea
# - Tao/nap vocab CSV tu train
# - Chi dung data_test de validate
# - Early stopping, toi da 100 epoch
# - So sanh embedding voi CBOW

# In[ ]:


import os
import re
import csv
import json
import random
import subprocess
import sys
from collections import Counter

import numpy as np

try:
    from underthesea import word_tokenize
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'underthesea'])
    from underthesea import word_tokenize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DIRS = [
    os.path.join(BASE_DIR, 'data_train', 'train', 'neg'),
    os.path.join(BASE_DIR, 'data_train', 'train', 'pos'),
]
TEST_DIRS = [
    os.path.join(BASE_DIR, 'data_test', 'test', 'neg'),
    os.path.join(BASE_DIR, 'data_test', 'test', 'pos'),
]

VOCAB_CSV = os.path.join(OUTPUT_DIR, 'vocab_train.csv')
CBOW_EMB_PATH = os.path.join(OUTPUT_DIR, 'cbow_embeddings.npy')
SKIPGRAM_EMB_PATH = os.path.join(OUTPUT_DIR, 'skipgram_embeddings.npy')

WINDOW_SIZE = 2
MIN_FREQ = 2
EMBED_DIM = 100
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 5
LR = 0.05
MAX_TRAIN_PAIRS = 300000
MAX_VAL_PAIRS = 70000


# In[ ]:


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_vi(text):
    text = normalize_text(text)
    tokenized = word_tokenize(text, format='text')
    return tokenized.split()

def read_sentences(dir_paths):
    sentences = []
    for folder in dir_paths:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith('.txt'):
                continue
            p = os.path.join(folder, name)
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                tokens = tokenize_vi(f.read())
                if len(tokens) >= 5:
                    sentences.append(tokens)
    return sentences

def create_vocab_csv(train_sentences, vocab_csv, min_freq=2):
    counter = Counter()
    for s in train_sentences:
        counter.update(s)

    vocab = ['<PAD>', '<UNK>'] + [w for w, c in counter.items() if c >= min_freq]

    with open(vocab_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'index', 'freq'])
        writer.writerow(['<PAD>', 0, 0])
        writer.writerow(['<UNK>', 1, 0])
        for i, w in enumerate(vocab[2:], start=2):
            writer.writerow([w, i, counter[w]])

def load_vocab_csv(vocab_csv):
    vocab = []
    with open(vocab_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r['index']))
        for r in rows:
            vocab.append(r['token'])
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_idx

def ensure_vocab_from_train(train_sentences, vocab_csv, min_freq=2):
    if os.path.exists(vocab_csv):
        print('Da ton tai vocab CSV, se nap lai:', vocab_csv)
    else:
        print('Chua co vocab CSV, tao moi tu train:', vocab_csv)
        create_vocab_csv(train_sentences, vocab_csv, min_freq=min_freq)
    return load_vocab_csv(vocab_csv)

def sentences_to_ids(sentences, word_to_idx):
    unk = word_to_idx['<UNK>']
    return [[word_to_idx.get(tok, unk) for tok in s] for s in sentences]

def generate_skipgram_pairs(sent_ids, window=2, max_pairs=None, seed=42):
    centers, contexts = [], []
    for sent in sent_ids:
        n = len(sent)
        for i in range(window, n - window):
            c = sent[i]
            around = sent[i-window:i] + sent[i+1:i+window+1]
            for w in around:
                centers.append(c)
                contexts.append(w)

    centers = np.array(centers, dtype=np.int32)
    contexts = np.array(contexts, dtype=np.int32)

    if max_pairs is not None and len(contexts) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(contexts), size=max_pairs, replace=False)
        centers = centers[idx]
        contexts = contexts[idx]
    return centers, contexts


# In[ ]:


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def eval_skipgram(E, W, b, centers, contexts, batch_size=512):
    n = len(contexts)
    total_loss, total_acc, steps = 0.0, 0.0, 0
    for i in range(0, n, batch_size):
        x = centers[i:i+batch_size]
        y = contexts[i:i+batch_size]
        h = E[x]
        probs = softmax(h @ W + b)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        acc = (np.argmax(probs, axis=1) == y).mean()
        total_loss += float(loss)
        total_acc += float(acc)
        steps += 1
    return total_loss / max(steps, 1), total_acc / max(steps, 1)

def train_skipgram_earlystop(centers_tr, contexts_tr, centers_val, contexts_val, vocab_size, embed_dim, epochs=100, patience=5, batch_size=512, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    W = rng.normal(0, 0.1, size=(embed_dim, vocab_size)).astype(np.float32)
    b = np.zeros(vocab_size, dtype=np.float32)

    best_val = float('inf')
    best_params = (E.copy(), W.copy(), b.copy())
    wait = 0

    n = len(contexts_tr)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        x_all = centers_tr[order]
        y_all = contexts_tr[order]

        total_loss, total_acc, steps = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            x = x_all[i:i+batch_size]
            y = y_all[i:i+batch_size]
            bs = len(y)

            h = E[x]
            logits = h @ W + b
            probs = softmax(logits)

            loss = -np.log(probs[np.arange(bs), y] + 1e-12).mean()
            acc = (np.argmax(probs, axis=1) == y).mean()

            dlogits = probs.copy()
            dlogits[np.arange(bs), y] -= 1.0
            dlogits /= bs

            dW = h.T @ dlogits
            db = dlogits.sum(axis=0)
            dh = dlogits @ W.T

            dE = np.zeros_like(E)
            np.add.at(dE, x, dh)

            E -= lr * dE
            W -= lr * dW
            b -= lr * db

            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1

        tr_loss = total_loss / max(steps, 1)
        tr_acc = total_acc / max(steps, 1)
        val_loss, val_acc = eval_skipgram(E, W, b, centers_val, contexts_val, batch_size=batch_size)

        print(f'Epoch {epoch:03d}/100 - train_loss: {tr_loss:.4f} - train_acc: {tr_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_params = (E.copy(), W.copy(), b.copy())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping tai epoch {epoch}, best val_loss = {best_val:.4f}')
                break

    return best_params

def normalize_matrix(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norm

def nearest_neighbors(word, emb, word_to_idx, top_k=5):
    if word not in word_to_idx:
        return []
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    emb_n = normalize_matrix(emb)
    idx = word_to_idx[word]
    sims = emb_n @ emb_n[idx]
    best = np.argsort(-sims)
    out = []
    for j in best:
        w = idx_to_word.get(int(j), '')
        if w in ('', '<PAD>', '<UNK>', word):
            continue
        out.append((w, float(sims[j])))
        if len(out) == top_k:
            break
    return out


# In[ ]:


train_sent = read_sentences(TRAIN_DIRS)
test_sent = read_sentences(TEST_DIRS)

vocab, word_to_idx = ensure_vocab_from_train(train_sent, VOCAB_CSV, min_freq=MIN_FREQ)

train_ids = sentences_to_ids(train_sent, word_to_idx)
test_ids = sentences_to_ids(test_sent, word_to_idx)

centers_tr, contexts_tr = generate_skipgram_pairs(train_ids, window=WINDOW_SIZE, max_pairs=MAX_TRAIN_PAIRS, seed=SEED)
centers_val, contexts_val = generate_skipgram_pairs(test_ids, window=WINDOW_SIZE, max_pairs=MAX_VAL_PAIRS, seed=SEED)

print(f'So cau train: {len(train_sent):,} | test: {len(test_sent):,}')
print(f'Vocab size: {len(vocab):,}')
print(f'Cap train: {len(contexts_tr):,} | cap val(test): {len(contexts_val):,}')

E_sg, W_sg, b_sg = train_skipgram_earlystop(
    centers_tr, contexts_tr, centers_val, contexts_val,
    vocab_size=len(vocab), embed_dim=EMBED_DIM,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR, seed=SEED
)


# In[ ]:


np.save(SKIPGRAM_EMB_PATH, E_sg)
print('Da luu skipgram embedding:', SKIPGRAM_EMB_PATH)

if os.path.exists(CBOW_EMB_PATH):
    E_cbow = np.load(CBOW_EMB_PATH)
    n = min(len(E_cbow), len(E_sg))
    ec = normalize_matrix(E_cbow[:n])
    es = normalize_matrix(E_sg[:n])
    cos_same = np.sum(ec * es, axis=1)
    print(f'Cosine trung binh cung mot tu (CBOW vs Skip-gram): {cos_same.mean():.4f}')

    for q in ['good', 'bad', 'movie', 'film', 'not']:
        if q in word_to_idx:
            print(f'\nTu: {q}')
            print(' CBOW    :', nearest_neighbors(q, E_cbow, word_to_idx, top_k=5))
            print(' Skipgram:', nearest_neighbors(q, E_sg, word_to_idx, top_k=5))
else:
    print('Khong tim thay CBOW embedding. Hay chay Bai 1 truoc de so sanh.')

