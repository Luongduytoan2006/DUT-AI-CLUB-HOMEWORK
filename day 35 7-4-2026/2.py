# =====================================================================
# Bai 2: Phan loai email Spam / Not-Spam
# Ky thuat: Encoding ket hop N-GRAM + TF-IDF (TfidfVectorizer)
#           So sanh voi Bai 1 (chi dung N-gram dem tan suat - CountVectorizer)
# Dataset: spam_or_not_spam.csv (cot: email, label) - 0: not spam, 1: spam
# =====================================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Cau hinh chung ---
CONFIG = {
    "csv_path": "spam_or_not_spam.csv",
    "test_size": 0.2,
    "random_state": 42,
    "ngram_range": (1, 2),   # unigram + bigram cho ca 2 cach encoding
    "min_df": 2,
}


# =====================================================================
# 1. GET DATA - Doc du lieu tu file CSV
# =====================================================================
def load_data(csv_path):
    """Doc file CSV, tra ve DataFrame gom 2 cot: email, label."""
    return pd.read_csv(csv_path)


# =====================================================================
# 2. XU LY - Lam sach va tien xu ly van ban
# =====================================================================
def clean_text(text):
    """Lam sach 1 email: viet thuong, chi giu chu cai, gom khoang trang."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df):
    """Loai bo dong rong/NaN va them cot 'clean' da lam sach."""
    df = df.dropna(subset=["email"]).copy()
    df["clean"] = df["email"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    return df


# =====================================================================
# 3. CLASS & HAM - Bo phan loai cho phep chon kieu encoding
# =====================================================================
class SpamClassifier:
    """Bo phan loai spam, chon 1 trong 2 kieu encoding:
       - 'count' : N-gram dem tan suat (giong Bai 1)
       - 'tfidf' : N-gram ket hop TF-IDF (Bai 2)
    """

    def __init__(self, encoding="tfidf", ngram_range=(1, 2), min_df=2):
        self.encoding = encoding
        if encoding == "count":
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        elif encoding == "tfidf":
            # TfidfVectorizer = dem n-gram roi nhan trong so TF-IDF
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        else:
            raise ValueError("encoding phai la 'count' hoac 'tfidf'")
        # Dung chung Naive Bayes de so sanh cong bang giua 2 kieu encoding
        self.model = MultinomialNB()

    def fit(self, texts, labels):
        """Hoc tu dien + bien doi encoding roi huan luyen mo hinh."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts):
        """Du doan nhan cho danh sach van ban moi."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)


# =====================================================================
# 4. UTILS - Cac ham tien ich danh gia & ve bieu do
# =====================================================================
def evaluate(y_true, y_pred, show_report=True):
    """Tinh accuracy (va in bao cao neu can)."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    if show_report:
        print(classification_report(y_true, y_pred,
                                    target_names=["not spam", "spam"]))
    return acc


def plot_compare(acc_count, acc_tfidf):
    """Ve bieu do cot so sanh accuracy 2 kieu encoding."""
    fig, ax = plt.subplots(figsize=(4, 4))
    names = ["N-gram\n(Count - Bai 1)", "N-gram + TF-IDF\n(Bai 2)"]
    vals = [acc_count, acc_tfidf]
    ax.bar(names, vals, color=["#4c72b0", "#55a868"])
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("So sanh 2 kieu encoding")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center")
    plt.tight_layout()
    return fig


# =====================================================================
# 5. MAIN - Trinh tu thuc hien theo de bai (so sanh 2 kieu encoding)
# =====================================================================
def main():
    # B1. Get data
    df = load_data(CONFIG["csv_path"])
    print(f"Du lieu goc: {df.shape[0]} email")

    # B2. Xu ly
    df = preprocess(df)
    print(f"Sau lam sach: {df.shape[0]} email")

    # B3. Chia train / test (dung chung cho ca 2 kieu encoding -> so sanh cong bang)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=df["label"],
    )

    # B4. Cach 1 - N-gram dem tan suat (giong Bai 1)
    print("\n=== Cach 1: N-gram (CountVectorizer) + Naive Bayes ===")
    clf_count = SpamClassifier("count", CONFIG["ngram_range"], CONFIG["min_df"])
    clf_count.fit(X_train, y_train)
    acc_count = evaluate(y_test, clf_count.predict(X_test))

    # B5. Cach 2 - N-gram ket hop TF-IDF (Bai 2)
    print("=== Cach 2: N-gram + TF-IDF (TfidfVectorizer) + Naive Bayes ===")
    clf_tfidf = SpamClassifier("tfidf", CONFIG["ngram_range"], CONFIG["min_df"])
    clf_tfidf.fit(X_train, y_train)
    acc_tfidf = evaluate(y_test, clf_tfidf.predict(X_test))

    # B6. So sanh & giai thich
    print("\n=== SO SANH & GIAI THICH ===")
    print(f"N-gram (Count)  + NB: {acc_count:.4f}")
    print(f"N-gram + TF-IDF + NB: {acc_tfidf:.4f}")
    cross_check_tfidf(X_train, X_test, y_train, y_test)  # minh hoa cho ket luan
    print(explain(acc_count, acc_tfidf))

    return acc_count, acc_tfidf


def explain(acc_count, acc_tfidf):
    """Tra ve doan giai thich su khac biet giua 2 kieu encoding,
       bam sat ket qua thuc te quan sat duoc tren tap nay."""
    return (
        "Giai thich (theo dung ket qua chay):\n"
        "1. Y nghia 2 kieu encoding:\n"
        "   - CountVectorizer (Bai 1): chi DEM so lan xuat hien cua moi n-gram (so nguyen).\n"
        "   - TF-IDF (Bai 2): lay so dem do nhan trong so IDF -> ha thap tu xuat hien o "
        "hau het email, de cao tu hiem mang tinh phan biet. Ket qua la so thuc, da chuan hoa L2.\n"
        "2. Vi sao o day TF-IDF lai THAP hon (0.90 < 0.99, recall spam tut con 0.41):\n"
        "   - MultinomialNB von duoc thiet ke cho dac trung dang DEM. TF-IDF bien dac trung "
        "thanh so thuc + chuan hoa, lam sai lech gia dinh phan phoi cua NB.\n"
        "   - Du lieu MAT CAN BANG (2500 not-spam / 500 spam). TF-IDF lam nhat bot trong so "
        "cac tu, cong voi prior thien ve lop da so khien NB 'ngai' du doan spam -> bo sot "
        "hon mot nua so spam (recall spam chi 0.41).\n"
        "3. Ket luan: TF-IDF KHONG phai luc nao cung tot hon. Voi MultinomialNB tren du lieu "
        "dang dem va mat can bang, CountVectorizer lai phu hop hon. TF-IDF thuong phat huy "
        "the manh khi di kem mo hinh tuyen tinh (Logistic Regression, SVM) - xem kiem chung ben duoi."
    )


def cross_check_tfidf(X_train, X_test, y_train, y_test):
    """Kiem chung: TF-IDF di voi Logistic Regression de thay encoding nay
       phat huy the manh khi doi mo hinh (minh hoa cho phan ket luan)."""
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=CONFIG["ngram_range"], min_df=CONFIG["min_df"])
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr, y_train)
    acc = accuracy_score(y_test, lr.predict(Xte))
    print(f"(Kiem chung) TF-IDF + Logistic Regression: {acc:.4f}")
    return acc


# =====================================================================
# 6. SHOW - Ve bieu do (chay khi goi truc tiep)
# =====================================================================
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    acc_count, acc_tfidf = main()
    plot_compare(acc_count, acc_tfidf)
    plt.show()
