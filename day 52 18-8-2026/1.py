import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class AudioSignalAnalyzer:
    def __init__(self, audio_path=None):
        self.audio_path = audio_path or librosa.example("trumpet")
        self.signal = None
        self.sample_rate = None

    def load_data(self):
        self.signal, self.sample_rate = librosa.load(self.audio_path, sr=None)
        return self.signal, self.sample_rate

    def print_audio_info(self):
        print(f"Sampling rate (Tan so lay mau): {self.sample_rate} Hz")
        print(f"Do dai mang du lieu: {len(self.signal)} samples")
        print(f"Thoi luong: {len(self.signal) / self.sample_rate:.2f} giay")

    def plot_waveform(self):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(
            self.signal,
            sr=self.sample_rate,
            color="royalblue",
        )
        plt.title("Waveform")
        plt.xlabel("Thoi gian (s)")
        plt.ylabel("Bien do")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_fft(self):
        number_of_samples = len(self.signal)
        fft_values = np.fft.rfft(self.signal)
        magnitude = np.abs(fft_values)
        frequencies = np.fft.rfftfreq(
            number_of_samples,
            d=1 / self.sample_rate,
        )

        plt.figure(figsize=(14, 5))
        plt.plot(frequencies, magnitude, color="darkorange")
        plt.title("Pho bien do (Magnitude Spectrum - FFT)")
        plt.xlabel("Tan so (Hz)")
        plt.ylabel("Bien do (Magnitude)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    analyzer = AudioSignalAnalyzer()
    analyzer.load_data()
    analyzer.print_audio_info()
    analyzer.plot_waveform()
    analyzer.plot_fft()


if __name__ == "__main__":
    main()
