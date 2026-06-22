import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import importlib

m1 = importlib.import_module("1")
RnnStack_Manual = m1.RnnStack_Manual
TextDataset_Manual = m1.TextDataset_Manual
load_split = m1.load_split
build_vocab = m1.build_vocab
PAD_TOKEN = m1.PAD_TOKEN

class SentimentClassifier_Manual(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, num_classes, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(0.3)
        self.rnn = RnnStack_Manual(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        output, h_n = self.rnn(emb)
        idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
        last_hidden = output.gather(1, idx).squeeze(1)
        return self.fc(self.dropout(last_hidden))

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            lens = batch['length'].to(device)
            labs = batch['label'].to(device)
            logits = model(ids, lens)
            loss_val = criterion(logits, labs).item()
            total_loss += loss_val * labs.size(0)
            preds = logits.argmax(1)
            for i in range(len(preds)):
                if preds[i] == labs[i]:
                    correct += 1
            total += labs.size(0)
    return total_loss / total, correct / total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_dir = Path.cwd()
    root = nb_dir.parent
    train_dir = root / 'data' / 'data_train' / 'train'
    val_dir = root / 'data' / 'data_train' / 'test'
    test_dir = root / 'data' / 'data_test' / 'test'
    
    train_texts, train_labels = load_split(train_dir)
    val_texts, val_labels = load_split(val_dir)
    test_texts, test_labels = load_split(test_dir)
    
    vocab_path = nb_dir / 'vocab.csv'
    word2id = build_vocab(train_texts, vocab_path)
    
    vocab_tokens = []
    for k in word2id.keys():
        vocab_tokens.append(k)
    vocab_size = len(vocab_tokens)
    
    train_ds = TextDataset_Manual(train_texts, train_labels, word2id)
    val_ds = TextDataset_Manual(val_texts, val_labels, word2id)
    test_ds = TextDataset_Manual(test_texts, test_labels, word2id)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = SentimentClassifier_Manual(
        vocab_size=vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        num_layers=1, 
        num_classes=2, 
        pad_idx=word2id[PAD_TOKEN]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        r_loss = 0.0
        r_correct = 0
        r_total = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            lens = batch['length'].to(device)
            labs = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(ids, lens)
            loss = criterion(logits, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            r_loss += loss.item() * labs.size(0)
            preds = logits.argmax(1)
            for i in range(len(preds)):
                if preds[i] == labs[i]:
                    r_correct += 1
            r_total += labs.size(0)
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch:02d} | Train Loss: {r_loss/r_total:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    if Path('best_model.pt').exists():
        model.load_state_dict(torch.load('best_model.pt'))
        
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Acc: {test_acc:.4f}')
