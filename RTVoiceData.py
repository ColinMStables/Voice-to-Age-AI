import torch
import torchaudio
import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import os
import pandas as pd

SAMPLE_RATE = 16000     # in Hz
BUFFER_DURATION = 5.0   # seconds kept in the buffor
CHUNK_DURATION = 0.25   # seconds per mic read
N_MELS = 64

BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

class RTVoice():

    def __init__(self, n_mel) -> None:

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mel
        )
        
        self.buffer = deque(torch.zeros(BUFFER_SIZE), maxlen=BUFFER_SIZE)

        self.use_normalized = True

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE
        )
        self.stream.start()

    def plot_waveform(self, waveform, ax):
        ax.clear()
        t = np.linspace(0, BUFFER_DURATION, waveform.shape[0])
        ax.plot(t, waveform, linewidth=1)
        ax.set_xlim([0, BUFFER_DURATION])
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Live Waveform")
        ax.grid(True)

    def plot_spectrogram(self, specgram, ax):
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
        ax.clear()
        spec_db = db_transform(specgram)
        ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title("Live Mel Spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")

    def collect_voice_data(self):
        chunk, _ = self.stream.read(CHUNK_SIZE)
        chunk_tensor = torch.from_numpy(chunk.squeeze()).clone()

        self.buffer.extend(chunk_tensor.tolist())
        audio_tensor = torch.tensor(list(self.buffer), dtype=torch.float32)

        mel_spec = self.mel_transform(audio_tensor)

        return mel_spec
    
    def plot_real_time_voice(self):

        plt.ion()
        _ , axs = plt.subplots(2, 1, figsize=(10, 6))
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
                mel_spec = self.mel_transform(audio_tensor)

                if(self.use_normalized):
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

                # Update plots
                self.plot_waveform(audio_tensor.numpy(), axs[0])
                self.plot_spectrogram(mel_spec, axs[1])

                plt.tight_layout()
                plt.pause(0.01)

        except KeyboardInterrupt:
            print("\nStopped streaming.")
            stream.stop()
            stream.close()
            plt.ioff()



if __name__ == "__main__":

    # TODO make a device selector

    print("Printing Device List")
    devices = sd.query_devices()

    for device in devices:
        print(f"Index: {device['index']} ---- Device name: {device['name']}")

    print("\nWhich device would you like to use? Use index number to select")

    device_index = input()

    try:
        device_index = int(device_index)
    except Exception as e:
        print(f"Ran into error converting device index to an integer {e}")

    sd.default.device = device_index
    
    voice = RTVoice(64)

    voice.plot_real_time_voice()
