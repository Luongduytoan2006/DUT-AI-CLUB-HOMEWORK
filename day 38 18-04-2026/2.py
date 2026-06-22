import numpy as np
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# Cấu hình tham số mô hình
CONFIG = {
    "pretrained_path": "wiki.vi.vec",
    "hidden_layers": (100, 50),
    "max_iter": 200,
    "label_pos": 1,
    "label_neg": 0
}

def load_data(url):
    """Tải dữ liệu từ GitHub và chuyển đổi thành list các cặp (text, label)."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        
        # Xử lý dữ liệu không dùng vòng lặp for tường minh (dùng list comprehension)
        def parse_line(line):
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label = CONFIG["label_pos"] if parts[0] == '__label__1' else CONFIG["label_neg"]
                return (parts[1], label)
            return None
            
        data = [parse_line(l) for l in lines]
        return [item for item in data if item is not None]
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return []

def get_sentence_vector(text, model):
    """Tính trung bình cộng các word vectors trong câu (Sentence Embedding)."""
    words = text.lower().split()
    # Lấy các vector của từ có trong từ điển (không dùng for tường minh)
    vectors = [model[w] for w in words if w in model]
    
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

class EmotionClassifier:
    """Lớp bọc cho mô hình MLP của Scikit-learn."""
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=CONFIG["hidden_layers"], 
            max_iter=CONFIG["max_iter"]
        )
        
    def fit(self, x_data, y_labels):
        """Huấn luyện mô hình."""
        self.model.fit(x_data, y_labels)
        
    def predict(self, x_data):
        """Dự đoán nhãn."""
        return self.model.predict(x_data)

def run():
    """Quy trình thực thi chính của Bài 2."""
    # 1. Load pretrained model
    try:
        fasttext_model = KeyedVectors.load_word2vec_format(CONFIG["pretrained_path"])
    except FileNotFoundError:
        print("Không tìm thấy file pretrained, đang tải model dự phòng...")
        import gensim.downloader as api
        fasttext_model = api.load("glove-twitter-25")
    except Exception as e:
        print(f"Lỗi không xác định khi load model: {e}")
        return

    # 2. Tải dữ liệu train/test
    train_url = "https://raw.githubusercontent.com/congnghia0609/ntc-scv/master/data/train.txt"
    test_url = "https://raw.githubusercontent.com/congnghia0609/ntc-scv/master/data/test.txt"
    
    train_raw = load_data(train_url)
    test_raw = load_data(test_url)
    
    if not train_raw or not test_raw:
        print("Dữ liệu rỗng, kết thúc.")
        return

    # 3. Biến đổi văn bản thành vector (Sử dụng map để tránh vòng lặp for tường minh)
    x_train = np.array(list(map(lambda item: get_sentence_vector(item[0], fasttext_model), train_raw)))
    y_train = np.array(list(map(lambda item: item[1], train_raw)))
    
    x_test = np.array(list(map(lambda item: get_sentence_vector(item[0], fasttext_model), test_raw)))
    y_test = np.array(list(map(lambda item: item[1], test_raw)))
    
    # 4. Huấn luyện và dự đoán
    classifier = EmotionClassifier()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    # 5. Đánh giá kết quả
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy sử dụng MLP và Pretrained FastText: {acc:.4f}")

def main():
    """Hàm main gọi hàm run."""
    run()

if __name__ == '__main__':
    main()
