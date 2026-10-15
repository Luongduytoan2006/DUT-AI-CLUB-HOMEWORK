from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "audio" / "speech.wav"
TARGET_SAMPLE_RATE = 16000


class SpeechSpectrogramAnalyzer:
    def __init__(self, audio_path=AUDIO_PATH, sample_rate=TARGET_SAMPLE_RATE):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.signal = None
        self.duration = None
        self.wideband = None
        self.narrowband = None

    def load_data(self):
        self.signal, self.sample_rate = librosa.load(
            self.audio_path,
            sr=self.sample_rate,
            mono=True,
        )
        self.duration = len(self.signal) / self.sample_rate
        print(f"File: {self.audio_path}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"So sample: {len(self.signal)}")
        print(f"Thoi luong: {self.duration:.2f} giay")

    def milliseconds_to_samples(self, milliseconds):
        return int(milliseconds * self.sample_rate / 1000)

    def calculate_power_spectrogram(self, window_ms):
        window_length = self.milliseconds_to_samples(window_ms)
        hop_length = window_length // 4
        n_fft = 1
        while n_fft < window_length:
            n_fft *= 2

        stft = librosa.stft(
            self.signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window="hann",
            center=False,
        )
        power = np.abs(stft) ** 2
        power_db = librosa.power_to_db(power, ref=np.max)
        return {
            "power_db": power_db,
            "window_ms": window_ms,
            "window_length": window_length,
            "hop_length": hop_length,
            "n_fft": n_fft,
        }

    def create_spectrograms(self):
        self.wideband = self.calculate_power_spectrogram(window_ms=5)
        self.narrowband = self.calculate_power_spectrogram(window_ms=50)
        for name, result in [
            ("WIDEBAND", self.wideband),
            ("NARROWBAND", self.narrowband),
        ]:
            print(f"=== {name} SPECTROGRAM ===")
            print(
                f"Window {result['window_ms']} ms = "
                f"{result['window_length']} samples"
            )
            print(f"Hop = {result['hop_length']} samples")
            print(f"n_fft = {result['n_fft']}")
            print(f"So frame = {result['power_db'].shape[1]}")

    def plot_results(self):
        figure, axes = plt.subplots(3, 1, figsize=(15, 13))
        librosa.display.waveshow(
            self.signal,
            sr=self.sample_rate,
            ax=axes[0],
            color="royalblue",
        )
        axes[0].set_title("Waveform")
        axes[0].set_xlabel("Thoi gian (s)")
        axes[0].set_ylabel("Bien do")

        for axis, result, title in [
            (axes[1], self.wideband, "Wideband Spectrogram - cua so 5 ms"),
            (axes[2], self.narrowband, "Narrowband Spectrogram - cua so 50 ms"),
        ]:
            image = librosa.display.specshow(
                result["power_db"],
                sr=self.sample_rate,
                hop_length=result["hop_length"],
                n_fft=result["n_fft"],
                x_axis="time",
                y_axis="hz",
                cmap="magma",
                ax=axis,
            )
            axis.set_ylim(0, 8000)
            axis.set_title(title)
            axis.set_xlabel("Thoi gian (s)")
            axis.set_ylabel("Tan so (Hz)")
            figure.colorbar(image, ax=axis, format="%+2.0f dB")

        plt.tight_layout()
        plt.show()


def main():
    analyzer = SpeechSpectrogramAnalyzer()
    analyzer.load_data()
    analyzer.create_spectrograms()
    analyzer.plot_results()


if __name__ == "__main__":
    main()
