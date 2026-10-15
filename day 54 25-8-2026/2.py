import copy
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
import torchaudio.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["yes", "no", "up", "down"]
LABEL_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASSES)}


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        classes,
        max_samples_per_class=250,
        target_sample_rate=16000,
    ):
        self.file_list = []
        self.labels = []
        self.target_sample_rate = target_sample_rate
        self.target_length = target_sample_rate
        dataset_dir = Path(root_dir) / "mini_speech_commands"

        for class_name in classes:
            pattern = os.path.join(dataset_dir, class_name, "*.wav")
            wav_files = sorted(glob.glob(pattern))[:max_samples_per_class]
            self.file_list.extend(wav_files)
            self.labels.extend([LABEL_TO_INDEX[class_name]] * len(wav_files))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio, _ = librosa.load(
            self.file_list[index],
            sr=self.target_sample_rate,
            mono=True,
        )
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if waveform.shape[1] < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.target_length - waveform.shape[1]),
            )
        else:
            waveform = waveform[:, : self.target_length]

        return waveform, self.labels[index]


class Raw1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class Audio2DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class AudioClassificationExperiment:
    def __init__(self):
        self.full_dataset = None
        self.train_loader = None
        self.validation_loader = None
        self.raw_transform = nn.Identity()
        self.linear_transform = nn.Sequential(
            transforms.Spectrogram(n_fft=512, hop_length=160),
            transforms.AmplitudeToDB(),
        )
        self.log_mel_transform = nn.Sequential(
            transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
                n_mels=80,
            ),
            transforms.AmplitudeToDB(),
        )
        self.results = {}

    def prepare_dataset_folder(self):
        url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "mini_speech_commands.zip"
        )
        zip_path = BASE_DIR / "mini_speech_commands.zip"
        extract_dir = BASE_DIR / "dataset_speech_commands"
        shared_dir = BASE_DIR.parent / "day 54 25-8-2026" / "dataset_speech_commands"

        if not extract_dir.exists() and shared_dir.exists():
            extract_dir = shared_dir
        if not zip_path.exists() and not extract_dir.exists():
            print("Dang tai Mini Speech Commands...")
            urllib.request.urlretrieve(url, zip_path)
        if not extract_dir.exists():
            print("Dang giai nen du lieu...")
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)
        return extract_dir

    def load_data(self, batch_size=32):
        extract_dir = self.prepare_dataset_folder()
        self.full_dataset = SpeechCommandsDataset(extract_dir, CLASSES)
        train_size = int(0.8 * len(self.full_dataset))
        validation_size = len(self.full_dataset) - train_size
        train_dataset, validation_dataset = random_split(
            self.full_dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(SEED),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        print(
            f"Tong so mau: {len(self.full_dataset)} "
            f"(Train: {train_size}, Val: {validation_size})"
        )

    def print_transform_shapes(self):
        sample_waveform, _ = self.full_dataset[0]
        print(f"Raw shape: {self.raw_transform(sample_waveform).shape}")
        print(f"Linear Spectrogram shape: {self.linear_transform(sample_waveform).shape}")
        print(f"Log-Mel Spectrogram shape: {self.log_mel_transform(sample_waveform).shape}")

    def train_model(self, model, audio_transform, epochs=50, learning_rate=0.001):
        model = model.to(DEVICE)
        audio_transform = audio_transform.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_losses = []
        validation_accuracies = []
        best_accuracy = -1.0
        best_state = copy.deepcopy(model.state_dict())
        best_targets = []
        best_predictions = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for waveforms, labels in self.train_loader:
                waveforms = waveforms.to(DEVICE)
                labels = labels.to(DEVICE)
                inputs = audio_transform(waveforms)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * waveforms.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            correct = 0
            targets = []
            predictions = []
            with torch.no_grad():
                for waveforms, labels in self.validation_loader:
                    waveforms = waveforms.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(audio_transform(waveforms))
                    batch_predictions = outputs.argmax(dim=1)
                    correct += (batch_predictions == labels).sum().item()
                    targets.extend(labels.cpu().numpy())
                    predictions.extend(batch_predictions.cpu().numpy())

            validation_accuracy = correct / len(self.validation_loader.dataset)
            validation_accuracies.append(validation_accuracy)
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state = copy.deepcopy(model.state_dict())
                best_targets = targets.copy()
                best_predictions = predictions.copy()

            print(
                f"Epoch {epoch + 1:02d}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Accuracy: {validation_accuracy:.4f}"
            )

        model.load_state_dict(best_state)
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        return {
            "losses": train_losses,
            "accuracies": validation_accuracies,
            "targets": best_targets,
            "predictions": best_predictions,
        }

    def run_experiments(self, epochs=50):
        experiments = [
            ("Raw 1D-CNN", Raw1DCNN(len(CLASSES)), self.raw_transform),
            (
                "Linear Spectrogram 2D-CNN",
                Audio2DCNN(len(CLASSES)),
                self.linear_transform,
            ),
            (
                "Log-Mel Spectrogram 2D-CNN",
                Audio2DCNN(len(CLASSES)),
                self.log_mel_transform,
            ),
        ]

        for name, model, audio_transform in experiments:
            print(f"\n=== {name} ===")
            self.results[name] = self.train_model(
                model,
                audio_transform,
                epochs=epochs,
            )

        print("\n=== BANG TONG HOP ===")
        for name, result in self.results.items():
            print(f"{name}: {max(result['accuracies']):.4f}")

    def plot_results(self):
        log_mel_result = self.results["Log-Mel Spectrogram 2D-CNN"]
        number_of_epochs = len(log_mel_result["accuracies"])
        epoch_numbers = range(1, number_of_epochs + 1)
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))

        for name, result in self.results.items():
            axes[0, 0].plot(epoch_numbers, result["accuracies"], label=name)
        axes[0, 0].set_title("Validation Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        matrix = confusion_matrix(
            log_mel_result["targets"],
            log_mel_result["predictions"],
            labels=range(len(CLASSES)),
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Confusion Matrix - Log-Mel 2D-CNN")
        axes[0, 1].set_xlabel("Nhan du doan")
        axes[0, 1].set_ylabel("Nhan that")

        sample_waveform, sample_label = self.full_dataset[0]
        time_axis = np.arange(sample_waveform.shape[1]) / 16000
        axes[0, 2].plot(time_axis, sample_waveform.squeeze().numpy())
        axes[0, 2].set_title(f"Raw Waveform - {CLASSES[sample_label]}")
        axes[0, 2].set_xlabel("Thoi gian (s)")
        axes[0, 2].set_ylabel("Bien do")

        linear_spectrogram = (
            self.linear_transform(sample_waveform.to(DEVICE))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        linear_image = axes[1, 0].imshow(
            linear_spectrogram,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[1, 0].set_title("Linear Spectrogram (dB)")
        figure.colorbar(linear_image, ax=axes[1, 0])

        log_mel_spectrogram = (
            self.log_mel_transform(sample_waveform.to(DEVICE))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        log_mel_image = axes[1, 1].imshow(
            log_mel_spectrogram,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[1, 1].set_title("Log-Mel Spectrogram (dB)")
        figure.colorbar(log_mel_image, ax=axes[1, 1])

        report = classification_report(
            log_mel_result["targets"],
            log_mel_result["predictions"],
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            zero_division=0,
        )
        axes[1, 2].axis("off")
        axes[1, 2].text(0, 1, report, family="monospace", va="top")
        axes[1, 2].set_title("Classification Report - Log-Mel")

        plt.tight_layout()
        plt.show()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    experiment = AudioClassificationExperiment()
    experiment.load_data()
    experiment.print_transform_shapes()
    experiment.run_experiments(epochs=50)
    experiment.plot_results()


if __name__ == "__main__":
    main()
