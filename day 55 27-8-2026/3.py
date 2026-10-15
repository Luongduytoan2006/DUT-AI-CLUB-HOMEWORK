import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchaudio
import torchaudio.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.nn.functional import pad


SEED = 42
SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent
TARGET_WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]


class MFCCSVMClassifier:
    def __init__(self):
        self.word_to_index = {
            word: index for index, word in enumerate(TARGET_WORDS)
        }
        self.mfcc_transform = transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=13,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
            },
        )
        self.scaler = StandardScaler()
        self.model = SVC(kernel="rbf", C=10, gamma="scale")
        self.train_data = None
        self.test_data = None
        self.x_test_scaled = None
        self.y_test = None

    def load_data(self, train_per_class=400, test_per_class=100):
        data_root = BASE_DIR / "speech_commands_data"
        os.makedirs(data_root, exist_ok=True)
        print("Dang tai hoac mo Speech Commands Dataset...")
        train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            str(data_root),
            download=True,
            subset="training",
        )
        test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            str(data_root),
            download=True,
            subset="testing",
        )
        self.train_data = self.filter_dataset(train_dataset, train_per_class)
        self.test_data = self.filter_dataset(test_dataset, test_per_class)
        print(f"So mau Train: {len(self.train_data)}")
        print(f"So mau Test: {len(self.test_data)}")

    def filter_dataset(self, dataset, max_per_class):
        filtered_data = []
        counts = {word: 0 for word in TARGET_WORDS}

        for wav_path in dataset._walker:
            label = os.path.basename(os.path.dirname(wav_path))
            if label not in TARGET_WORDS or counts[label] >= max_per_class:
                continue

            audio, _ = librosa.load(
                wav_path,
                sr=SAMPLE_RATE,
                mono=True,
            )
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            filtered_data.append(
                (waveform, self.word_to_index[label])
            )
            counts[label] += 1

            if all(count == max_per_class for count in counts.values()):
                break

        return filtered_data

    def preprocess_and_extract_mfcc(self, dataset, max_length=16000):
        features = []
        labels = []

        for waveform, label in dataset:
            if waveform.shape[1] < max_length:
                waveform = pad(
                    waveform,
                    (0, max_length - waveform.shape[1]),
                )
            else:
                waveform = waveform[:, :max_length]

            mfcc = self.mfcc_transform(waveform)
            feature_vector = mfcc.mean(dim=-1).squeeze(0)
            features.append(feature_vector.numpy())
            labels.append(label)

        return np.array(features), np.array(labels)

    def train(self):
        x_train, y_train = self.preprocess_and_extract_mfcc(self.train_data)
        x_test, self.y_test = self.preprocess_and_extract_mfcc(self.test_data)
        x_train_scaled = self.scaler.fit_transform(x_train)
        self.x_test_scaled = self.scaler.transform(x_test)

        print(f"Kich thuoc X_train: {x_train.shape}")
        print(f"Kich thuoc X_test: {x_test.shape}")
        self.model.fit(x_train_scaled, y_train)

    def evaluate(self):
        predictions = self.model.predict(self.x_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(
            classification_report(
                self.y_test,
                predictions,
                labels=range(len(TARGET_WORDS)),
                target_names=TARGET_WORDS,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            self.y_test,
            predictions,
            labels=range(len(TARGET_WORDS)),
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_WORDS,
            yticklabels=TARGET_WORDS,
        )
        plt.title("Confusion Matrix - MFCC + SVM")
        plt.xlabel("Nhan du doan")
        plt.ylabel("Nhan that")
        plt.tight_layout()
        plt.show()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    classifier = MFCCSVMClassifier()
    classifier.load_data()
    classifier.train()
    classifier.evaluate()


if __name__ == "__main__":
    main()
