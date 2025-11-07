import torch
import torchaudio
import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque

# Parameters
SAMPLE_RATE = 16000     # Hz
BUFFER_DURATION = 5.0   # seconds kept in rolling buffer
CHUNK_DURATION = 0.25   # seconds per mic read
N_MELS = 128

BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# === Torchaudio transforms ===
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

def plot_waveform(waveform, sr, ax):
    ax.clear()
    t = np.linspace(0, BUFFER_DURATION, waveform.shape[0])
    ax.plot(t, waveform, linewidth=1)
    ax.set_xlim([0, BUFFER_DURATION])
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live Waveform")
    ax.grid(True)

def plot_spectrogram(specgram, ax):
    ax.clear()
    spec_db = db_transform(specgram)
    ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title("Live Mel Spectrogram")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")

def plot_real_time_voice():
    print("Starting real-time audio stream (Ctrl+C to stop)...")

    # Rolling buffer initialized with zeros
    buffer = deque(torch.zeros(BUFFER_SIZE), maxlen=BUFFER_SIZE)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE
    )
    stream.start()

    try:
        while True:
            # Read next chunk
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk_tensor = torch.from_numpy(chunk.squeeze()).clone()

            # Update rolling buffer
            buffer.extend(chunk_tensor.tolist())
            audio_tensor = torch.tensor(list(buffer), dtype=torch.float32)

            # Compute Mel spectrogram
            mel_spec = mel_transform(audio_tensor)
            #mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            # Update plots
            plot_waveform(audio_tensor.numpy(), SAMPLE_RATE, axs[0])
            plot_spectrogram(mel_spec, axs[1])

            plt.tight_layout()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped streaming.")
        stream.stop()
        stream.close()
        plt.ioff()
        plt.show()

class RTVoice():

    def __init__(self, n_mel) -> None:

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mel
        )
        
        self.buffer = deque(torch.zeros(BUFFER_SIZE), maxlen=BUFFER_SIZE)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE
        )
        self.stream.start()

    def collect_voice_data(self):
        chunk, _ = self.stream.read(CHUNK_SIZE)
        chunk_tensor = torch.from_numpy(chunk.squeeze()).clone()

        self.buffer.extend(chunk_tensor.tolist())
        audio_tensor = torch.tensor(list(self.buffer), dtype=torch.float32)

        mel_spec = self.mel_transform(audio_tensor)

        return mel_spec

if __name__ == "__main__":
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    sd.default.device = 6
    plot_real_time_voice()
