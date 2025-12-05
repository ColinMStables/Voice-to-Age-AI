import torch
import torchaudio as ta
import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import time

import NeuralNet

SAMPLE_RATE = 48000     # in Hz
BUFFER_DURATION = 5.0   # seconds kept in the buffer
CHUNK_DURATION = 0.25   # seconds per mic read
N_MELS = 64

BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['eighties', 'fifties', 'fourties', 'nineties', 'seventies', 'sixties', 'teens', 'thirties', 'twenties']
id_to_class = {i : cls for i,cls in enumerate(classes)}

class RTVoice():

    def __init__(self, n_mel) -> None:

        self.mel_transform = ta.transforms.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc = 52,
                log_mels=True,
                melkwargs={
                "n_fft" : 512,
                "hop_length" : 512,
                "n_mels" : n_mel,
                }
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
        db_transform = ta.transforms.AmplitudeToDB(stype="power")
        ax.clear()
        spec_db = db_transform(specgram)
        ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title("Live Mel Spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")

    def plot_pitch(self, pitch, ax):
        ax.clear()
        ax.plot(pitch, linewidth=1)
        ax.set_title("Pitch (Hz)")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(True)

    def plot_magnitude(self, magnitude, ax):
        ax.clear()
        mag_db = torch.log1p(magnitude)
        ax.imshow(mag_db.numpy(), origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title("Magnitude Spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Frequency bins")

    def collect_voice_data(self):
        chunk, _ = self.stream.read(CHUNK_SIZE)
        chunk_tensor = torch.from_numpy(chunk.squeeze()).clone()

        self.buffer.extend(chunk_tensor.tolist())
        audio_tensor = torch.tensor(list(self.buffer), dtype=torch.float32)

        mel_spec = self.mel_transform(audio_tensor)

        return mel_spec
    
    def run_real_time_age_detection(self, network, plot = False):

        if plot:
            plt.ion()
            _ , axs = plt.subplots(4, 1, figsize=(10, 6))

        buffer = deque(torch.zeros(BUFFER_SIZE), maxlen=BUFFER_SIZE)

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE
        )
        stream.start()

        while True:

                # Read next chunk
                chunk, _ = stream.read(CHUNK_SIZE)
                chunk_tensor = torch.from_numpy(chunk.squeeze()).clone()

                # Update rolling buffer
                buffer.extend(chunk_tensor.tolist())
                audio_tensor = torch.tensor(list(buffer), dtype=torch.float32)

                # Compute Mel spectrogram
                mel_spec = self.mel_transform(audio_tensor)

                pitch = ta.functional.detect_pitch_frequency(audio_tensor, SAMPLE_RATE)
                pitch = pitch.unsqueeze(0).unsqueeze(0) 
                print(pitch.shape)
                pitch = torch.nn.functional.interpolate(
                    pitch,
                    size=mel_spec.shape[-1],
                    mode="linear",
                    align_corners=False
                ).squeeze(0)

                pitch = torch.log1p(pitch)

                stft = torch.stft(
                    audio_tensor,
                    n_fft=10,
                    hop_length=512,
                    win_length=10,
                    window=torch.hann_window(10),
                    return_complex=True
                )
                magnitude = stft.abs()

                mel_spec = torch.cat([mel_spec, pitch], dim=0)
                mel_spec = torch.cat([mel_spec, magnitude], dim=0)

                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

                print(mel_spec.shape)
                
                time_before = time.time()
                print(id_to_class[network(mel_spec).argmax().item()])
                print(f"Time taken for inference: {time.time() - time_before} seconds")

                if plot:
                    # Update plots
                    self.plot_waveform(audio_tensor.cpu().numpy(), axs[0])
                    self.plot_spectrogram(mel_spec.cpu().squeeze(), axs[1])
                    self.plot_pitch(pitch.squeeze().cpu().numpy(), axs[2])
                    self.plot_magnitude(magnitude.cpu().squeeze(), axs[3])

                    plt.tight_layout()
                    plt.pause(0.01)


    
    def plot_real_time_voice(self):

        plt.ion()
        _ , axs = plt.subplots(4, 1, figsize=(10, 6))
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

                pitch = ta.functional.detect_pitch_frequency(audio_tensor, SAMPLE_RATE)
                pitch = pitch.unsqueeze(0).unsqueeze(0) 
                pitch = torch.nn.functional.interpolate(
                    pitch,
                    size=mel_spec.shape[-1],
                    mode="linear",
                    align_corners=False
                ).squeeze()

                stft = torch.stft(
                    audio_tensor,
                    n_fft=10,
                    hop_length=512,
                    win_length=10,
                    window=torch.hann_window(10),
                    return_complex=True
                )
                magnitude = stft.abs()  

                if(self.use_normalized):
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

                # Update plots
                self.plot_waveform(audio_tensor.numpy(), axs[0])
                self.plot_spectrogram(mel_spec, axs[1])
                self.plot_pitch(pitch.numpy(), axs[2])
                self.plot_magnitude(magnitude, axs[3])

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

    network = NeuralNet.Audio_Transformer(n_mels=59, classes=9, d_model=64, nheads=4, N=6, frame_length=500)
    network.load_state_dict(torch.load("./Max_Accuracy_Model.pth"))

    voice.run_real_time_age_detection(network, False)
