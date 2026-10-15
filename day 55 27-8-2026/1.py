from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "audio" / "speech.wav"


class AudioFeatureAnalyzer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.signal = None
        self.duration = None
        self.stft_results = {}

    @staticmethod
    def frame_signal(signal, frame_length, hop_length):
        if len(signal) < frame_length:
            signal = np.pad(signal, (0, frame_length - len(signal)))

        number_of_frames = 1 + (len(signal) - frame_length) // hop_length
        starts = np.arange(number_of_frames) * hop_length
        indices = starts[:, None] + np.arange(frame_length)[None, :]
        return signal[indices], starts

    def manual_stft(self, signal, frame_length, hop_length, n_fft=None):
        n_fft = n_fft or frame_length
        frames, starts = self.frame_signal(signal, frame_length, hop_length)
        windowed_frames = frames * np.hanning(frame_length)
        spectrum = np.fft.rfft(windowed_frames, n=n_fft, axis=1).T
        frequencies = np.fft.rfftfreq(n_fft, d=1 / self.sample_rate)
        frame_times = (starts + frame_length / 2) / self.sample_rate
        return spectrum, frequencies, frame_times

    def create_synthetic_signal(self):
        duration = 2.0
        time_axis = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        signal = np.where(
            time_axis < 1.0,
            np.sin(2 * np.pi * 440 * time_axis),
            np.sin(2 * np.pi * 880 * time_axis),
        ).astype(np.float32)
        return signal, time_axis

    def compare_fft_and_stft(self):
        signal, time_axis = self.create_synthetic_signal()
        fft_values = np.fft.rfft(signal)
        fft_magnitude = np.abs(fft_values)
        fft_frequencies = np.fft.rfftfreq(
            len(signal),
            d=1 / self.sample_rate,
        )

        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        stft, _, _ = self.manual_stft(signal, frame_length, hop_length)
        stft_db = 20 * np.log10(np.maximum(np.abs(stft), 1e-10))
        stft_db -= stft_db.max()

        figure, axes = plt.subplots(3, 1, figsize=(14, 11))
        axes[0].plot(time_axis, signal, color="royalblue")
        axes[0].axvline(1.0, color="red", linestyle="--")
        axes[0].set_title("Tin hieu 440 Hz chuyen sang 880 Hz")
        axes[0].set_xlabel("Thoi gian (s)")
        axes[0].set_ylabel("Bien do")

        axes[1].plot(fft_frequencies, fft_magnitude, color="darkorange")
        axes[1].set_xlim(0, 1500)
        axes[1].set_title("FFT tren toan bo tin hieu")
        axes[1].set_xlabel("Tan so (Hz)")
        axes[1].set_ylabel("Magnitude")
        axes[1].grid(alpha=0.3)

        image = librosa.display.specshow(
            stft_db,
            sr=self.sample_rate,
            hop_length=hop_length,
            n_fft=frame_length,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=axes[2],
        )
        axes[2].set_ylim(0, 1500)
        axes[2].set_title("STFT tu cai dat")
        figure.colorbar(image, ax=axes[2], format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def load_data(self, audio_path=AUDIO_PATH):
        self.signal, self.sample_rate = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
        )
        self.duration = len(self.signal) / self.sample_rate
        print(f"File: {audio_path}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"So sample: {len(self.signal)}")
        print(f"Thoi luong: {self.duration:.2f} giay")
        return self.signal, self.sample_rate

    def calculate_window_stfts(self, window_sizes_ms=(10, 25, 50)):
        for window_ms in window_sizes_ms:
            frame_length = int(self.sample_rate * window_ms / 1000)
            hop_length = frame_length // 2
            spectrum, frequencies, frame_times = self.manual_stft(
                self.signal,
                frame_length,
                hop_length,
            )
            magnitude = np.abs(spectrum)
            power = magnitude**2
            db_spectrogram = 10 * np.log10(np.maximum(power, 1e-10))
            db_spectrogram -= db_spectrogram.max()

            self.stft_results[window_ms] = {
                "frame_length": frame_length,
                "hop_length": hop_length,
                "magnitude": magnitude,
                "power": power,
                "db": db_spectrogram,
                "frequencies": frequencies,
                "times": frame_times,
            }
            print(
                f"{window_ms:2d} ms: frame={frame_length:3d}, "
                f"frames={magnitude.shape[1]:3d}, "
                f"bins={magnitude.shape[0]:3d}, "
                f"delta_f={self.sample_rate / frame_length:.1f} Hz"
            )

    def plot_window_stfts(self):
        figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        for axis, (window_ms, result) in zip(axes, self.stft_results.items()):
            image = axis.imshow(
                result["db"],
                origin="lower",
                aspect="auto",
                extent=[0, self.duration, 0, self.sample_rate / 2],
                cmap="magma",
                vmin=-80,
                vmax=0,
            )
            axis.set_ylim(0, 8000)
            axis.set_ylabel("Tan so (Hz)")
            axis.set_title(f"Cua so {window_ms} ms")
            figure.colorbar(image, ax=axis, format="%+2.0f dB")

        axes[-1].set_xlabel("Thoi gian (s)")
        plt.tight_layout()
        plt.show()

    def extract_and_plot_features(self):
        n_fft = 512
        win_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        speech_stft, _, _ = self.manual_stft(
            self.signal,
            win_length,
            hop_length,
            n_fft,
        )
        speech_power = np.abs(speech_stft) ** 2
        speech_db = librosa.power_to_db(speech_power, ref=np.max)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=64,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)

        figure, axes = plt.subplots(4, 1, figsize=(14, 16))
        librosa.display.waveshow(
            self.signal,
            sr=self.sample_rate,
            ax=axes[0],
            color="royalblue",
        )
        axes[0].set_title("1. Waveform")

        plots = [
            (speech_db, "hz", "2. Power Spectrogram (dB)"),
            (log_mel, "mel", "3. Log-Mel Spectrogram"),
            (mfcc, None, "4. MFCC (13 he so)"),
        ]
        for axis, (data, y_axis, title) in zip(axes[1:], plots):
            image = librosa.display.specshow(
                data,
                sr=self.sample_rate,
                hop_length=hop_length,
                x_axis="time",
                y_axis=y_axis,
                cmap="magma",
                ax=axis,
            )
            axis.set_title(title)
            figure.colorbar(image, ax=axis)

        plt.tight_layout()
        plt.show()

    def compare_with_librosa(self):
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        hann_window = np.hanning(frame_length)
        manual_spectrum, _, _ = self.manual_stft(
            self.signal,
            frame_length,
            hop_length,
            frame_length,
        )
        librosa_spectrum = librosa.stft(
            self.signal,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            window=hann_window,
            center=False,
        )
        difference = np.abs(
            np.abs(manual_spectrum) - np.abs(librosa_spectrum)
        )
        print(f"Sai khac lon nhat: {difference.max():.8f}")
        print(f"Sai khac trung binh: {difference.mean():.8f}")

        manual_db = librosa.amplitude_to_db(np.abs(manual_spectrum), ref=np.max)
        librosa_db = librosa.amplitude_to_db(np.abs(librosa_spectrum), ref=np.max)
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        for axis, data, title in [
            (axes[0], manual_db, "STFT tu cai dat"),
            (axes[1], librosa_db, "librosa.stft"),
            (axes[2], difference, "Sai khac tuyet doi"),
        ]:
            image = axis.imshow(data, origin="lower", aspect="auto", cmap="magma")
            axis.set_title(title)
            axis.set_xlabel("Frame thoi gian")
            axis.set_ylabel("Bin tan so")
            figure.colorbar(image, ax=axis)
        plt.tight_layout()
        plt.show()


def main():
    analyzer = AudioFeatureAnalyzer()
    analyzer.compare_fft_and_stft()
    analyzer.load_data()
    analyzer.calculate_window_stfts()
    analyzer.plot_window_stfts()
    analyzer.extract_and_plot_features()
    analyzer.compare_with_librosa()


if __name__ == "__main__":
    main()
