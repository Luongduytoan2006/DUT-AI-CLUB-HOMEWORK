import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioCNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SpeechCommandsClassifier:
    def __init__(self):
        self.url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "mini_speech_commands.zip"
        )
        self.zip_path = BASE_DIR / "mini_speech_commands.zip"
        self.extract_dir = BASE_DIR / "dataset_speech_commands"
        self.target_classes = ["go", "stop", "yes", "no"]
        self.model = AudioCNN1D(len(self.target_classes)).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = None
        self.test_loader = None
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

    def prepare_dataset_folder(self):
        shared_dir = BASE_DIR.parent / "day 54 25-8-2026" / "dataset_speech_commands"
        if not self.extract_dir.exists() and shared_dir.exists():
            self.extract_dir = shared_dir

        if not self.zip_path.exists() and not self.extract_dir.exists():
            print("Dang tai Mini Speech Commands...")
            urllib.request.urlretrieve(self.url, self.zip_path)

        if not self.extract_dir.exists():
            print("Dang giai nen du lieu...")
            with zipfile.ZipFile(self.zip_path, "r") as zip_file:
                zip_file.extractall(self.extract_dir)

        return self.extract_dir / "mini_speech_commands"

    def load_data(self, max_samples_per_class=150):
        dataset_dir = self.prepare_dataset_folder()
        audio_samples = []
        labels = []

        for label, class_name in enumerate(self.target_classes):
            pattern = os.path.join(dataset_dir, class_name, "*.wav")
            wav_files = sorted(glob.glob(pattern))
            for wav_path in wav_files[:max_samples_per_class]:
                audio, _ = librosa.load(
                    wav_path,
                    sr=16000,
                    mono=True,
                    duration=1.0,
                )
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                audio_samples.append(audio[:16000])
                labels.append(label)

        print(f"Tong so mau da nap: {len(audio_samples)}")
        return np.array(audio_samples, dtype=np.float32), np.array(labels)

    def create_data_loaders(self, audio_samples, labels, batch_size=16):
        audio_samples = np.expand_dims(audio_samples, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(
            audio_samples,
            labels,
            test_size=0.2,
            random_state=SEED,
            stratify=labels,
        )

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_dataset = TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        print(f"Train shape: {x_train.shape}")
        print(f"Test shape: {x_test.shape}")

    def run_epoch(self, data_loader, training):
        self.model.train(training)
        total_loss = 0.0
        correct = 0

        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        return total_loss / len(data_loader.dataset), correct / len(data_loader.dataset)

    def train(self, epochs=25):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.run_epoch(self.train_loader, True)
            test_loss, test_accuracy = self.run_epoch(self.test_loader, False)

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["test_accuracy"].append(test_accuracy)

            print(
                f"Epoch {epoch + 1:02d}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
            )

    def predict_test_data(self):
        predictions = []
        labels = []
        self.model.eval()

        with torch.no_grad():
            for inputs, batch_labels in self.test_loader:
                outputs = self.model(inputs.to(DEVICE))
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                labels.extend(batch_labels.numpy())

        return np.array(labels), np.array(predictions)

    def evaluate(self):
        true_labels, predictions = self.predict_test_data()
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, self.history["test_loss"], label="Test Loss")
        axes[0].set_title("Loss theo Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, self.history["train_accuracy"], label="Train Accuracy")
        axes[1].plot(epochs, self.history["test_accuracy"], label="Test Accuracy")
        axes[1].set_title("Accuracy theo Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(
            classification_report(
                true_labels,
                predictions,
                target_names=self.target_classes,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.target_classes,
            yticklabels=self.target_classes,
        )
        plt.title("Confusion Matrix - AudioCNN1D")
        plt.xlabel("Nhan du doan")
        plt.ylabel("Nhan that")
        plt.tight_layout()
        plt.show()


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pipeline = SpeechCommandsClassifier()
    audio_samples, labels = pipeline.load_data()
    pipeline.create_data_loaders(audio_samples, labels)
    pipeline.train(epochs=25)
    pipeline.evaluate()


if __name__ == "__main__":
    main()
