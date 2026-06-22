#!/usr/bin/env python
# coding: utf-8


# In[6]:


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
CBOW_MAP_PATH = os.path.join(OUTPUT_DIR, 'cbow_word_to_idx.json')

WINDOW_SIZE = 2
MIN_FREQ = 2
EMBED_DIM = 100
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 5
LR = 0.05
MAX_TRAIN_PAIRS = 250000
MAX_VAL_PAIRS = 60000


# In[7]:


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_vi(text):
    text = normalize_text(text)
    tokenized = word_tokenize(text, format='text')
    tokens = tokenized.split()
    return tokens

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

    vocab = ['<PAD>', '<UNK>']
    vocab += [w for w, c in counter.items() if c >= min_freq]

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

def generate_cbow_pairs(sent_ids, window=2, max_pairs=None, seed=42):
    contexts, targets = [], []
    for sent in sent_ids:
        n = len(sent)
        for i in range(window, n - window):
            contexts.append(sent[i-window:i] + sent[i+1:i+window+1])
            targets.append(sent[i])

    contexts = np.array(contexts, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    if max_pairs is not None and len(targets) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(targets), size=max_pairs, replace=False)
        contexts = contexts[idx]
        targets = targets[idx]
    return contexts, targets


# In[8]:


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def eval_cbow(E, W, b, contexts, targets, batch_size=512):
    n = len(targets)
    total_loss, total_acc, steps = 0.0, 0.0, 0
    for i in range(0, n, batch_size):
        x = contexts[i:i+batch_size]
        y = targets[i:i+batch_size]
        emb = E[x]
        h = emb.mean(axis=1)
        probs = softmax(h @ W + b)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        acc = (np.argmax(probs, axis=1) == y).mean()
        total_loss += float(loss)
        total_acc += float(acc)
        steps += 1
    return total_loss / max(steps, 1), total_acc / max(steps, 1)

def train_cbow_earlystop(contexts_tr, targets_tr, contexts_val, targets_val, vocab_size, embed_dim, epochs=100, patience=5, batch_size=512, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    W = rng.normal(0, 0.1, size=(embed_dim, vocab_size)).astype(np.float32)
    b = np.zeros(vocab_size, dtype=np.float32)

    best_val = float('inf')
    best_params = (E.copy(), W.copy(), b.copy())
    wait = 0

    n = len(targets_tr)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        x_all = contexts_tr[order]
        y_all = targets_tr[order]

        total_loss, total_acc, steps = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            x = x_all[i:i+batch_size]
            y = y_all[i:i+batch_size]
            bs = len(y)

            emb = E[x]
            h = emb.mean(axis=1)
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
            dh_each = dh / x.shape[1]
            for k in range(x.shape[1]):
                np.add.at(dE, x[:, k], dh_each)

            E -= lr * dE
            W -= lr * dW
            b -= lr * db

            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1

        tr_loss = total_loss / max(steps, 1)
        tr_acc = total_acc / max(steps, 1)
        val_loss, val_acc = eval_cbow(E, W, b, contexts_val, targets_val, batch_size=batch_size)

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


# In[ ]:


train_sent = read_sentences(TRAIN_DIRS)
test_sent = read_sentences(TEST_DIRS)

vocab, word_to_idx = ensure_vocab_from_train(train_sent, VOCAB_CSV, min_freq=MIN_FREQ)

train_ids = sentences_to_ids(train_sent, word_to_idx)
test_ids = sentences_to_ids(test_sent, word_to_idx)

contexts_tr, targets_tr = generate_cbow_pairs(train_ids, window=WINDOW_SIZE, max_pairs=MAX_TRAIN_PAIRS, seed=SEED)
contexts_val, targets_val = generate_cbow_pairs(test_ids, window=WINDOW_SIZE, max_pairs=MAX_VAL_PAIRS, seed=SEED)

print(f'So cau train: {len(train_sent):,} | test: {len(test_sent):,}')
print(f'Vocab size: {len(vocab):,}')
print(f'Cap train: {len(targets_tr):,} | cap val(test): {len(targets_val):,}')

E, W, b = train_cbow_earlystop(
    contexts_tr, targets_tr, contexts_val, targets_val,
    vocab_size=len(vocab), embed_dim=EMBED_DIM,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR, seed=SEED
)


# In[ ]:


np.save(CBOW_EMB_PATH, E)
with open(CBOW_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

print('Da luu:')
print('-', CBOW_EMB_PATH)
print('-', CBOW_MAP_PATH)
print('-', VOCAB_CSV)
print('Embedding shape:', E.shape)

