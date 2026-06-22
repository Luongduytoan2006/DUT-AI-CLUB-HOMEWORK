import re
import torch
import requests
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt

# Định nghĩa các hằng số cấu hình
CONFIG = {
    "lr": 0.001,
    "batch_size": 32,
    "embed_dim": 100,
    "n_gram_min": 3,
    "n_gram_max": 6,
    "epochs": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

NUM_CLASSES = 2
LABEL_MAP = {"__label__1": 1, "__label__0": 0}

def load_data(url):
    """Tải dữ liệu từ URL và phân tách label, text."""
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    data = []
    for line in lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            label = LABEL_MAP.get(parts[0], 0)
            data.append((parts[1], label))
    return data

def clean_text(text):
    """Làm sạch văn bản: viết thường và xóa ký tự đặc biệt."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_ngrams(word, n_min, n_max):
    """Tạo các n-gram ký tự cho một từ."""
    word = f"<{word}>"
    res = []
    for n in range(n_min, n_max + 1):
        for i in range(len(word) - n + 1):
            res.append(word[i:i+n])
    return res

class FastTextModel(nn.Module):
    """Mô hình FastText: EmbeddingBag -> Linear."""
    def __init__(self, vocab_size, embed_dim, output_dim):
        super(FastTextModel, self).__init__()
        # EmbeddingBag thực hiện tính trung bình các vector embedding tự động
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, text, offsets):
        # Trả về kết quả sau khi đi qua lớp tuyến tính
        return self.fc(self.embedding(text, offsets))

def transform(text, vocab):
    """Chuyển đổi văn bản thành danh sách các index của n-gram."""
    idx = []
    for word in clean_text(text).split():
        for ng in get_ngrams(word, CONFIG["n_gram_min"], CONFIG["n_gram_max"]):
            idx.append(vocab.get(ng, 0))
    return idx

def collate_fn(batch, vocab):
    """Hàm chuẩn bị batch cho DataLoader."""
    labels, texts, offsets = [], [], [0]
    for text, label in batch:
        labels.append(label)
        indices = torch.tensor(transform(text, vocab), dtype=torch.int64)
        texts.append(indices)
        offsets.append(indices.size(0))
    # Chuyển đổi sang Tensor và đưa lên thiết bị (CPU/GPU)
    text_tensor = torch.cat(texts).to(CONFIG["device"])
    label_tensor = torch.tensor(labels).to(CONFIG["device"])
    offset_tensor = torch.tensor(offsets[:-1]).cumsum(0).to(CONFIG["device"])
    return text_tensor, label_tensor, offset_tensor

def run():
    """Quy trình huấn luyện và đánh giá mô hình."""
    # 1. Tải dữ liệu
    train_data = load_data("https://raw.githubusercontent.com/congnghia0609/ntc-scv/master/data/train.txt")
    test_data = load_data("https://raw.githubusercontent.com/congnghia0609/ntc-scv/master/data/test.txt")
    
    # 2. Xây dựng Vocabulary từ n-grams
    all_ngrams = []
    for text, _ in train_data:
        for word in clean_text(text).split():
            all_ngrams.extend(get_ngrams(word, CONFIG["n_gram_min"], CONFIG["n_gram_max"]))
    
    counts = Counter(all_ngrams)
    # Chỉ giữ các n-gram xuất hiện nhiều hơn 1 lần
    vocab = {ng: i + 1 for i, (ng, c) in enumerate(counts.items()) if c > 1}
    vocab['<UNK>'] = 0
    
    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(
        train_data, batch_size=CONFIG["batch_size"], 
        shuffle=True, collate_fn=lambda b: collate_fn(b, vocab)
    )
    
    # 4. Khởi tạo mô hình, loss và optimizer
    model = FastTextModel(len(vocab), CONFIG["embed_dim"], NUM_CLASSES).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    # 5. Vòng lặp huấn luyện
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for texts, labels, offsets in train_loader:
            optimizer.zero_grad()
            outputs = model(texts, offsets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {total_loss/len(train_loader):.4f}")

    # 6. Đánh giá trên tập test
    model.eval()
    test_loader = DataLoader(
        test_data, batch_size=CONFIG["batch_size"], 
        collate_fn=lambda b: collate_fn(b, vocab)
    )
    correct = 0
    with torch.no_grad():
        for texts, labels, offsets in test_loader:
            preds = model(texts, offsets).argmax(1)
            correct += (preds == labels).sum().item()
    
    accuracy = correct / len(test_data)
    print(f"Final Test Accuracy: {accuracy:.4f}")

def main():
    """Hàm main gọi hàm run."""
    run()

if __name__ == '__main__':
    main()
