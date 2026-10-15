import glob
import urllib.request
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.mixture import GaussianMixture


SEED = 42
SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent


class GMMVoiceActivityDetector:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.signal = None
        self.speech_intervals = []
        self.features = None
        self.frame_times = None
        self.true_labels = None
        self.predictions = None
        self.llr = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.noise_model = GaussianMixture(n_components=2, random_state=SEED)
        self.speech_model = GaussianMixture(n_components=3, random_state=SEED)

    def prepare_dataset(self):
        url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "mini_speech_commands.zip"
        )
        zip_path = BASE_DIR / "mini_speech_commands.zip"
        extract_dir = BASE_DIR / "dataset_speech_commands"

        if not zip_path.exists() and not extract_dir.exists():
            print("Dang tai Mini Speech Commands...")
            urllib.request.urlretrieve(url, zip_path)

        if not extract_dir.exists():
            print("Dang giai nen du lieu...")
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)

        wav_files = glob.glob(
            str(extract_dir / "**" / "*.wav"),
            recursive=True,
        )
        return sorted(wav_files)

    def create_test_signal(self, wav_files):
        random_generator = np.random.RandomState(SEED)
        selected_files = random_generator.choice(wav_files, 4, replace=False)
        speech_signals = []

        for wav_path in selected_files:
            audio, _ = librosa.load(
                wav_path,
                sr=self.sample_rate,
                mono=True,
            )
            speech_signals.append(audio.astype(np.float32))

        initial_noise = random_generator.normal(
            0,
            0.0008,
            int(self.sample_rate * 0.5),
        ).astype(np.float32)
        signal_parts = [initial_noise]
        current_time = 0.5

        for speech in speech_signals:
            start_time = current_time
            end_time = start_time + len(speech) / self.sample_rate
            self.speech_intervals.append((start_time, end_time))
            signal_parts.append(speech)

            noise_gap = random_generator.normal(
                0,
                0.0008,
                int(self.sample_rate * 0.4),
            ).astype(np.float32)
            signal_parts.append(noise_gap)
            current_time = end_time + 0.4

        self.signal = np.concatenate(signal_parts).astype(np.float32)
        print(f"Tan so lay mau: {self.sample_rate} Hz")
        print(f"Thoi luong tin hieu: {len(self.signal) / self.sample_rate:.2f} giay")
        print(f"Cac khoang co giong noi: {self.speech_intervals}")

    def extract_subband_features(self, frame_length_ms=20, hop_length_ms=10):
        frame_length = int(self.sample_rate * frame_length_ms / 1000)
        hop_length = int(self.sample_rate * hop_length_ms / 1000)
        power_spectrogram = np.abs(
            librosa.stft(
                self.signal,
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length,
            )
        ) ** 2
        frequencies = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=frame_length,
        )
        bands = [
            (80, 250),
            (250, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 3000),
            (3000, 4000),
        ]

        number_of_frames = power_spectrogram.shape[1]
        self.features = np.zeros(
            (number_of_frames, len(bands)),
            dtype=np.float32,
        )
        for band_index, (minimum, maximum) in enumerate(bands):
            frequency_indices = np.where(
                (frequencies >= minimum) & (frequencies <= maximum)
            )[0]
            band_power = np.sum(
                power_spectrogram[frequency_indices, :],
                axis=0,
            )
            self.features[:, band_index] = np.log10(band_power + 1e-10)

        self.frame_times = librosa.frames_to_time(
            np.arange(number_of_frames),
            sr=self.sample_rate,
            hop_length=hop_length,
        )
        self.true_labels = np.zeros(number_of_frames, dtype=int)
        for frame_index, frame_time in enumerate(self.frame_times):
            for start_time, end_time in self.speech_intervals:
                if start_time <= frame_time <= end_time:
                    self.true_labels[frame_index] = 1
                    break

        print(f"Kich thuoc dac trung: {self.features.shape}")
        print(f"So khung speech: {np.sum(self.true_labels)}")

    def train_and_predict(self, threshold=0.0):
        noise_features = self.features[self.frame_times < 0.4]
        total_energy = np.sum(self.features, axis=1)
        speech_threshold = np.percentile(total_energy, 60)
        speech_features = self.features[total_energy > speech_threshold]

        self.noise_model.fit(noise_features)
        self.speech_model.fit(speech_features)

        noise_log_probability = self.noise_model.score_samples(self.features)
        speech_log_probability = self.speech_model.score_samples(self.features)
        self.llr = speech_log_probability - noise_log_probability
        self.predictions = (self.llr > threshold).astype(int)

    def evaluate(self):
        print("=== Classification Report ===")
        print(
            classification_report(
                self.true_labels,
                self.predictions,
                target_names=["Noise (0)", "Speech (1)"],
                zero_division=0,
            )
        )
        self.fpr, self.tpr, _ = roc_curve(self.true_labels, self.llr)
        self.roc_auc = auc(self.fpr, self.tpr)
        print(f"ROC-AUC Score: {self.roc_auc:.4f}")

    def plot_results(self):
        figure = plt.figure(figsize=(14, 10))
        time_axis = np.arange(len(self.signal)) / self.sample_rate

        waveform_axis = figure.add_subplot(2, 2, 1)
        waveform_axis.plot(time_axis, self.signal, color="blue", alpha=0.6)
        for start_time, end_time in self.speech_intervals:
            waveform_axis.axvspan(start_time, end_time, color="green", alpha=0.2)
        waveform_axis.set_title("Waveform va Ground Truth")
        waveform_axis.set_xlabel("Time (s)")
        waveform_axis.set_ylabel("Amplitude")

        llr_axis = figure.add_subplot(2, 2, 2)
        llr_axis.plot(self.frame_times, self.llr, color="orange", label="LLR")
        llr_axis.axhline(0.0, color="red", linestyle="--", label="Threshold")
        llr_axis.fill_between(
            self.frame_times,
            self.llr.min(),
            self.llr.max(),
            where=self.predictions == 1,
            color="green",
            alpha=0.3,
        )
        llr_axis.set_title("GMM VAD")
        llr_axis.set_xlabel("Time (s)")
        llr_axis.set_ylabel("LLR")
        llr_axis.legend()

        matrix_axis = figure.add_subplot(2, 2, 3)
        matrix = confusion_matrix(self.true_labels, self.predictions)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Noise", "Speech"],
            yticklabels=["Noise", "Speech"],
            ax=matrix_axis,
        )
        matrix_axis.set_title("Confusion Matrix")
        matrix_axis.set_xlabel("Predicted Label")
        matrix_axis.set_ylabel("True Label")

        roc_axis = figure.add_subplot(2, 2, 4)
        roc_axis.plot(
            self.fpr,
            self.tpr,
            color="darkorange",
            label=f"ROC (AUC = {self.roc_auc:.4f})",
        )
        roc_axis.plot([0, 1], [0, 1], color="navy", linestyle="--")
        roc_axis.set_title("ROC Curve")
        roc_axis.set_xlabel("False Positive Rate")
        roc_axis.set_ylabel("True Positive Rate")
        roc_axis.legend(loc="lower right")

        plt.tight_layout()
        plt.show()


def main():
    detector = GMMVoiceActivityDetector()
    wav_files = detector.prepare_dataset()
    detector.create_test_signal(wav_files)
    detector.extract_subband_features()
    detector.train_and_predict()
    detector.evaluate()
    detector.plot_results()


if __name__ == "__main__":
    main()
