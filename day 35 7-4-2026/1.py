# =====================================================================
# Bai 1: Phan loai email Spam / Not-Spam
# Ky thuat: Tien xu ly -> Tokenization -> Encoding bang N-GRAM (CountVectorizer)
#           -> Mo hinh Machine Learning (Naive Bayes & Logistic Regression)
# Dataset: spam_or_not_spam.csv (cot: email, label) - 0: not spam, 1: spam
# =====================================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Cau hinh chung ---
CONFIG = {
    "csv_path": "spam_or_not_spam.csv",  # file giai nen tu archive.zip
    "test_size": 0.2,                    # ti le tap test
    "random_state": 42,                  # co dinh de tai lap ket qua
    "ngram_range": (1, 2),               # dung unigram + bigram (n-gram)
    "min_df": 2,                         # bo cac tu xuat hien < 2 lan
}


# =====================================================================
# 1. GET DATA - Doc du lieu tu file CSV
# =====================================================================
def load_data(csv_path):
    """Doc file CSV, tra ve DataFrame gom 2 cot: email, label."""
    df = pd.read_csv(csv_path)
    return df


# =====================================================================
# 2. XU LY - Lam sach va tien xu ly van ban
# =====================================================================
def clean_text(text):
    """Lam sach 1 email: viet thuong, xoa ky tu dac biet, gom khoang trang."""
    text = str(text).lower()              # dua ve chu thuong
    text = re.sub(r"[^a-z\s]", " ", text)  # chi giu chu cai va khoang trang
    text = re.sub(r"\s+", " ", text)      # gom nhieu khoang trang thanh 1
    return text.strip()


def preprocess(df):
    """Loai bo dong rong/NaN va them cot 'clean' da lam sach."""
    df = df.dropna(subset=["email"]).copy()   # bo dong email bi thieu
    df["clean"] = df["email"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]        # bo dong rong sau khi lam sach
    return df


# =====================================================================
# 3. CLASS & HAM - Bo phan loai Spam dung N-gram + ML
# =====================================================================
class SpamClassifier:
    """Bo phan loai spam: Encoding bang N-gram (CountVectorizer) + mo hinh ML."""

    def __init__(self, ngram_range=(1, 2), min_df=2, model=None):
        # Vectorizer chuyen van ban -> ma tran dem tan suat cac n-gram
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        # Mac dinh dung Naive Bayes (rat hop voi bai toan dem tu)
        self.model = model if model is not None else MultinomialNB()

    def fit(self, texts, labels):
        """Hoc tu dien n-gram tu tap train roi huan luyen mo hinh."""
        X = self.vectorizer.fit_transform(texts)   # fit + transform tap train
        self.model.fit(X, labels)
        return self

    def predict(self, texts):
        """Du doan nhan cho danh sach van ban moi."""
        X = self.vectorizer.transform(texts)       # chi transform (dung vocab da hoc)
        return self.model.predict(X)

    def vocab_size(self):
        """So luong dac trung (n-gram) trong tu dien."""
        return len(self.vectorizer.vocabulary_)


# =====================================================================
# 4. UTILS - Cac ham tien ich danh gia & ve bieu do
# =====================================================================
def evaluate(y_true, y_pred):
    """In bao cao va tra ve accuracy."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=["not spam", "spam"]))
    return acc


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    """Ve ma tran nham lan (confusion matrix)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not spam", "spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not spam", "spam"])
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te"); ax.set_title(title)
    # ghi so luong len tung o
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="red", fontsize=14)
    plt.tight_layout()
    return fig


# =====================================================================
# 5. MAIN - Trinh tu thuc hien theo de bai
# =====================================================================
def main():
    # B1. Get data
    df = load_data(CONFIG["csv_path"])
    print(f"Du lieu goc: {df.shape[0]} email")

    # B2. Xu ly / tien xu ly
    df = preprocess(df)
    print(f"Sau lam sach: {df.shape[0]} email")
    print("Phan bo nhan:\n", df["label"].value_counts())

    # B3. Chia train / test
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=df["label"],   # giu nguyen ti le spam/not-spam o 2 tap
    )

    # B4. Huan luyen voi N-gram encoding + Naive Bayes
    clf = SpamClassifier(ngram_range=CONFIG["ngram_range"],
                         min_df=CONFIG["min_df"],
                         model=MultinomialNB())
    clf.fit(X_train, y_train)
    print(f"\nSo dac trung n-gram: {clf.vocab_size()}")

    # B5. Danh gia
    print("\n=== Ket qua N-gram + Naive Bayes ===")
    y_pred = clf.predict(X_test)
    evaluate(y_test, y_pred)

    # B6. Thu them mo hinh Logistic Regression de so sanh
    clf_lr = SpamClassifier(ngram_range=CONFIG["ngram_range"],
                            min_df=CONFIG["min_df"],
                            model=LogisticRegression(max_iter=1000))
    clf_lr.fit(X_train, y_train)
    print("=== Ket qua N-gram + Logistic Regression ===")
    evaluate(y_test, clf_lr.predict(X_test))

    return y_test, y_pred


# =====================================================================
# 6. SHOW - Ve bieu do (chay khi goi truc tiep)
# =====================================================================
if __name__ == "__main__":
    # doi thu muc lam viec ve dung folder chua file nay
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    y_test, y_pred = main()
    plot_confusion(y_test, y_pred, title="Bai 1: N-gram + Naive Bayes")
    plt.show()
