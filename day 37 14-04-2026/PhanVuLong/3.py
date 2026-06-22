#!/usr/bin/env python
# coding: utf-8

# # Bai 3 - Phan loai cam xuc (NumPy + underthesea)
# 
# Yeu cau da dap ung:
# - Tien xu ly bang underthesea
# - Tao/nap vocab CSV tu train
# - Chi dung data_test de validate
# - Early stopping, toi da 100 epoch

# In[ ]:


import os
import re
import csv
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
TRAIN_NEG = os.path.join(BASE_DIR, 'data_train', 'train', 'neg')
TRAIN_POS = os.path.join(BASE_DIR, 'data_train', 'train', 'pos')
TEST_NEG = os.path.join(BASE_DIR, 'data_test', 'test', 'neg')
TEST_POS = os.path.join(BASE_DIR, 'data_test', 'test', 'pos')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
VOCAB_CSV = os.path.join(OUTPUT_DIR, 'vocab_train.csv')

MIN_FREQ = 2
MAX_LEN = 200
EMBED_DIM = 48
H1 = 64
H2 = 32
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 8
LR = 0.01


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

def load_labeled_data(neg_dir, pos_dir):
    texts, labels = [], []
    for name in os.listdir(neg_dir):
        if name.endswith('.txt'):
            with open(os.path.join(neg_dir, name), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(tokenize_vi(f.read()))
                labels.append(0)
    for name in os.listdir(pos_dir):
        if name.endswith('.txt'):
            with open(os.path.join(pos_dir, name), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(tokenize_vi(f.read()))
                labels.append(1)
    return texts, np.array(labels, dtype=np.float32)

def create_vocab_csv(train_tokenized, vocab_csv, min_freq=2):
    counter = Counter()
    for tks in train_tokenized:
        counter.update(tks)
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

def ensure_vocab_from_train(train_tokenized, vocab_csv, min_freq=2):
    if os.path.exists(vocab_csv):
        print('Da ton tai vocab CSV, se nap lai:', vocab_csv)
    else:
        print('Chua co vocab CSV, tao moi tu train:', vocab_csv)
        create_vocab_csv(train_tokenized, vocab_csv, min_freq=min_freq)
    return load_vocab_csv(vocab_csv)

def tokens_to_sequences(tokenized_texts, word_to_idx):
    unk = word_to_idx['<UNK>']
    out = []
    for tks in tokenized_texts:
        out.append([word_to_idx.get(t, unk) for t in tks])
    return out

def pad_sequences_numpy(sequences, max_len=200, pad_value=0):
    x = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[:max_len]
        x[i, :len(trunc)] = trunc
    return x


# In[ ]:


def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

def bce(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()

def acc(y_true, y_prob):
    pred = (y_prob >= 0.5).astype(np.float32)
    return (pred == y_true).mean()

def init_params(vocab_size, max_len, embed_dim, h1, h2, seed=42):
    rng = np.random.default_rng(seed)
    return {
        'E': rng.normal(0, 0.05, size=(vocab_size, embed_dim)).astype(np.float32),
        'W1': rng.normal(0, 0.05, size=(max_len * embed_dim, h1)).astype(np.float32),
        'b1': np.zeros((1, h1), dtype=np.float32),
        'W2': rng.normal(0, 0.05, size=(h1, h2)).astype(np.float32),
        'b2': np.zeros((1, h2), dtype=np.float32),
        'W3': rng.normal(0, 0.05, size=(h2, 1)).astype(np.float32),
        'b3': np.zeros((1, 1), dtype=np.float32)
    }

def forward(x_idx, params):
    E = params['E']
    emb = E[x_idx]
    flat = emb.reshape(emb.shape[0], -1)

    z1 = flat @ params['W1'] + params['b1']
    a1 = relu(z1)
    z2 = a1 @ params['W2'] + params['b2']
    a2 = relu(z2)
    z3 = a2 @ params['W3'] + params['b3']
    p = sigmoid(z3)
    cache = (x_idx, emb, flat, z1, a1, z2, a2, p)
    return p, cache

def backward(y_true, cache, params):
    x_idx, emb, flat, z1, a1, z2, a2, p = cache
    bs = y_true.shape[0]

    dz3 = (p - y_true) / bs
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0, keepdims=True)

    da2 = dz3 @ params['W3'].T
    dz2 = da2 * relu_grad(z2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ params['W2'].T
    dz1 = da1 * relu_grad(z1)
    dW1 = flat.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    dflat = dz1 @ params['W1'].T
    demb = dflat.reshape(emb.shape)

    dE = np.zeros_like(params['E'])
    for t in range(x_idx.shape[1]):
        np.add.at(dE, x_idx[:, t], demb[:, t, :])

    return {'E': dE, 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

def update(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]

def train_earlystop(X_tr, y_tr, X_val, y_val, params, epochs=100, patience=8, batch_size=128, lr=0.01, seed=42):
    rng = np.random.default_rng(seed)
    y_tr = y_tr.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    best_val = float('inf')
    best_params = {k: v.copy() for k, v in params.items()}
    wait = 0

    n = X_tr.shape[0]
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        Xs = X_tr[idx]
        ys = y_tr[idx]

        t_loss, t_acc, steps = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]

            p, cache = forward(xb, params)
            loss = bce(yb, p)
            a = acc(yb, p)

            grads = backward(yb, cache, params)
            update(params, grads, lr)

            t_loss += float(loss)
            t_acc += float(a)
            steps += 1

        p_val, _ = forward(X_val, params)
        val_loss = bce(y_val, p_val)
        val_acc = acc(y_val, p_val)
        print(f'Epoch {epoch:03d}/100 - train_loss: {t_loss/steps:.4f} - train_acc: {t_acc/steps:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_params = {k: v.copy() for k, v in params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping tai epoch {epoch}, best val_loss = {best_val:.4f}')
                break

    return best_params


# In[ ]:


train_tokens, y_train = load_labeled_data(TRAIN_NEG, TRAIN_POS)
test_tokens, y_test = load_labeled_data(TEST_NEG, TEST_POS)

vocab, word_to_idx = ensure_vocab_from_train(train_tokens, VOCAB_CSV, min_freq=MIN_FREQ)

X_train_seq = tokens_to_sequences(train_tokens, word_to_idx)
X_test_seq = tokens_to_sequences(test_tokens, word_to_idx)

X_train = pad_sequences_numpy(X_train_seq, max_len=MAX_LEN, pad_value=0)
X_test = pad_sequences_numpy(X_test_seq, max_len=MAX_LEN, pad_value=0)

print(f'So mau train: {len(X_train):,}')
print(f'So mau test (val): {len(X_test):,}')
print(f'Kich thuoc vocab: {len(vocab):,}')
print('Shape X_train:', X_train.shape)
print('Shape X_test :', X_test.shape)


# In[ ]:


params = init_params(len(vocab), MAX_LEN, EMBED_DIM, H1, H2, seed=SEED)
params = train_earlystop(
    X_train, y_train,
    X_test, y_test,
    params=params,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR, seed=SEED
)


# In[ ]:


p_test, _ = forward(X_test, params)
test_loss = bce(y_test.reshape(-1, 1), p_test)
test_acc = acc(y_test.reshape(-1, 1), p_test)

preds = (p_test.reshape(-1) >= 0.5).astype(np.int32)
y_int = y_test.astype(np.int32)
tn = int(np.sum((preds == 0) & (y_int == 0)))
fp = int(np.sum((preds == 1) & (y_int == 0)))
fn = int(np.sum((preds == 0) & (y_int == 1)))
tp = int(np.sum((preds == 1) & (y_int == 1)))

print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')
print('Confusion matrix (TN, FP, FN, TP):', (tn, fp, fn, tp))


# In[ ]:


def predict_sentiment(sentences, params, word_to_idx, max_len=200):
    tokenized = [tokenize_vi(s) for s in sentences]
    seqs = tokens_to_sequences(tokenized, word_to_idx)
    x = pad_sequences_numpy(seqs, max_len=max_len, pad_value=0)
    p, _ = forward(x, params)
    p = p.reshape(-1)
    out = []
    for s, prob in zip(sentences, p):
        label = 'positive' if prob >= 0.5 else 'negative'
        out.append((s, float(prob), label))
    return out

new_sentences = [
    'Bo phim rat hay va cam dong',
    'Noi dung qua te va dien xuat rat do',
    'Xem cung duoc, khong qua xuat sac'
]

for sent, prob, label in predict_sentiment(new_sentences, params, word_to_idx, max_len=MAX_LEN):
    print(f'\nSentence: {sent}')
    print(f'P(positive) = {prob:.4f} -> Predicted: {label}')

