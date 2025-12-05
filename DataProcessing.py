"""

Processes the mcv dataset into the mel spectrograms TODO Make this nicer

"""
import os
import csv
import torchaudio as ta
import torch
import pickle
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import multiprocessing
import psutil
import time


PATH = "./cv-corpus-23.0-2025-09-05/en/"

if(torch.cuda.is_available()):
    torch.set_default_device("cuda:0")

def remove_silence(waveform, sample_rate, frame_ms=30, hop_ms=10, energy_threshold=1e-4):
    """
    Remove silent parts of waveform based on short-time energy.
    
    waveform: torch.Tensor (1, num_samples)
    sample_rate: int
    frame_ms: frame length in milliseconds
    hop_ms: hop length in milliseconds
    energy_threshold: below this, frame is considered silent
    """
    frame_len = int(frame_ms / 1000 * sample_rate)
    hop_len = int(hop_ms / 1000 * sample_rate)
    
    # Pad waveform to make full frames
    pad_len = (waveform.shape[1] - frame_len) % hop_len
    if pad_len > 0:
        waveform = torch.cat([waveform, torch.zeros((1, hop_len - pad_len), device=waveform.device)], dim=1)
    
    # Unfold into frames
    frames = waveform.unfold(1, frame_len, hop_len)  # shape: (1, num_frames, frame_len)
    
    # Compute frame energy
    energy = torch.sum(frames**2, dim=2) / frame_len  # average energy per frame
    mask = energy > energy_threshold  # keep frames above threshold
    
    # Reconstruct waveform
    frames = frames.squeeze(0)[mask.squeeze(0)]
    if len(frames) == 0:
        return waveform  # if all silent, return original
    
    waveform_trimmed = frames.reshape(-1)
    return waveform_trimmed.unsqueeze(0)


def lpc_coefficients(waveform, order=12):

    x = waveform.squeeze(0).cpu().numpy()

    r = np.correlate(x, x, mode='full')
    r = r[len(r)//2:]
    if np.any(np.isnan(r)) or np.any(np.isinf(r)):
        return None
    R = toeplitz(r[:order])
    rhs = -r[1:order+1]
    a = np.linalg.solve(R, rhs)
    a = np.concatenate(([1], a))
    return a

def get_formants(waveform, sample_rate, lpc_order=50):
    x = waveform.squeeze(0).cpu().numpy()
    
    # Skip silent frames
    if np.mean(x**2) < 1e-6:
        return np.zeros(3)

    # Compute LPC
    a = lpc_coefficients(waveform, order=lpc_order)
    if a is None:
        return np.zeros(3)

    # Roots
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) > 0]  # positive freq

    angz = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angz * (sample_rate / (2 * np.pi))

    # Keep only realistic formants
    freqs = [f for f in freqs]
    freqs = np.sort(freqs)

    # Take first 3 or pad
    if len(freqs) < 3:
        freqs = np.pad(freqs, (0, 3-len(freqs)), 'constant')
    else:
        freqs = freqs[:3]

    return freqs



class DatasetProcessing():

    def __init__(self,
                 path,
                 process_id,
                 total_processes
                 ) -> None:
        self.path = path
        self.process_id = process_id
        self.total_processes = total_processes


    def get_ages(self, 
                 file : str
                 ):
        
        if(file[-4:] != ".tsv"):
            file = file + ".tsv"

        number_of_age = {}

        with open(self.path + file, "r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            tsv_reader.__next__()
            
            for row in tsv_reader:
                if(row[7] != ''):
                    if(not (row[7] in number_of_age.keys())):
                        number_of_age[row[7]] = 1
                    else:
                        number_of_age[row[7]] += 1

        print(number_of_age)

    def create_mel_transforms(
            self,
            csv_text,
            n_mels
            ):
        
        if(not os.path.isdir(self.path  + csv_text)):
            os.mkdir(self.path + csv_text)
        
        with open(self.path + csv_text + ".tsv", "r", encoding="utf-8") as f:
            last_line = f.seek(0, os.SEEK_END)
            f.seek(int((self.total_processes - self.process_id) / last_line), os.SEEK_SET) 

            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            tsv_reader.__next__()
            
            for ii, row in enumerate(tsv_reader):

                if(ii == int((self.total_processes - (self.process_id + 1)) / last_line)):
                    break

                if(row[7] != ""):
                    if(not os.path.isdir(self.path  + csv_text + "/" + row[7])):
                        os.mkdir(self.path + csv_text + "/" + row[7])
                    file_name = row[1]
                    
                    waveform, sample_rate = ta.load(self.path + "clips/" + file_name)
                    mel_transform = ta.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=1024,
                        hop_length=512,
                        n_mels=n_mels
                    )
                    mel_spec = mel_transform(waveform)

                    with open(self.path + csv_text + "/" + row[7] + "/" + file_name[:-4], "xb") as r:
                        pickle.dump(mel_spec, r)
    
    def create_mel_c(
            self,
            csv_text,
            n_mels,
            n_mfcc,
            name = ""
            ):
        
        if(not os.path.isdir(self.path  + csv_text + name)):
            os.mkdir(self.path + csv_text +name)

        tsv_size = 0
        with open(self.path + csv_text + ".tsv", "r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            next(tsv_reader)
            for ii, row in enumerate(tsv_reader):
                tsv_size += 1

        print(f"TSV size: {tsv_size}")
        with open(self.path + csv_text + ".tsv", "r", encoding="utf-8") as f:
            last_line = f.seek(0, os.SEEK_END)
            print(f"Process {self.process_id}: last line is {last_line}")
            f.seek(0, os.SEEK_SET)
            
            line_to_go = int(((self.total_processes) - self.process_id)/(self.total_processes + 1)*tsv_size)

            print(f"Process {self.process_id}: setting position to {line_to_go}")

            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            next(tsv_reader)

            for i in range(line_to_go):
                next(tsv_reader)
            
            last_line_used = int(((self.total_processes) - (self.process_id - 1))/(self.total_processes + 1)*tsv_size)
            print(f"Process {self.process_id}: going to {last_line_used}")
            for ii, row in enumerate(tsv_reader):
                
                if(ii == last_line_used - line_to_go):
                    break
            
                if(row[7] != ""):
                    if(not os.path.isdir(self.path  + csv_text + name + "/" + row[7])):
                        os.mkdir(self.path + csv_text + name + "/" + row[7])
                    file_name = row[1]
                    
                    if(os.path.isfile(self.path + csv_text + name + "/" + row[7] + "/" + file_name[:-4])):
                        continue

                    audio, sr = sf.read(self.path + "clips/" + file_name)
                    waveform = torch.tensor(audio).float().unsqueeze(0).cuda()
                    sample_rate = sr

                    original_waveform = waveform.clone()

                    waveform = remove_silence(waveform, sample_rate, frame_ms=30, hop_ms=10, energy_threshold=1e-4)

                    mel_transform = ta.transforms.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc = n_mfcc,
                        log_mels=True,
                        melkwargs={
                        "n_fft" : 1024,
                        "hop_length" : 512,
                        "n_mels" : n_mels,
                        }
                    ).to("cuda:0")

                    mel_spec = mel_transform(waveform)

                    pitch = ta.functional.detect_pitch_frequency(original_waveform, sample_rate)
                    pitch = pitch.unsqueeze(0)
                    pitch = torch.nn.functional.interpolate(pitch, size=mel_spec.shape[-1], mode="linear", align_corners=False)

                    pitch = torch.log1p(pitch)

                    window = torch.hann_window(1024, device=waveform.device)
                    stft = torch.stft(
                        waveform,
                        n_fft=1024,
                        hop_length=512,
                        win_length=1024,
                        window=window,
                        return_complex=True
                    )

                    magnitude = stft.abs() 

                    transform_spectral_centroid = ta.transforms.SpectralCentroid(sample_rate, n_fft=1024,hop_length=512)  # (1, 1, time)
                    spectral_centroid = transform_spectral_centroid(waveform).unsqueeze(0)
                    
                    # magnitude: (channel, freq_bins, time)
                    mag = magnitude + 1e-8  # avoid div by zero
                    freqs = torch.linspace(0, sample_rate/2, mag.shape[1]).to(mag.device)

                    # Spectral centroid
                    centroid = torch.sum(freqs[None, :, None] * mag, dim=1) / torch.sum(mag, dim=1)

                    # Spectral bandwidth
                    bandwidth = torch.sqrt(
                        torch.sum(((freqs[None, :, None] - centroid[:, None, :])**2) * mag, dim=1) / torch.sum(mag, dim=1)
                    )

                    # Spectral rolloff (approximate 85%)
                    cumsum_mag = torch.cumsum(mag, dim=1)
                    rolloff = torch.zeros_like(centroid)
                    threshold = 0.85 * torch.sum(mag, dim=1)
                    for i in range(mag.shape[2]):
                        rolloff[:, i] = freqs[(cumsum_mag[:, :, i] >= threshold[:, i]).nonzero()[0,1]]

                    # Spectral flatness
                    flatness = torch.exp(torch.mean(torch.log(mag), dim=1)) / torch.mean(mag, dim=1)


                    frame_len = int(0.025 * sample_rate)
                    hop_len   = int(0.010 * sample_rate)

                    frames = waveform.squeeze(0).cpu().unfold(0, frame_len, hop_len)
                    formants_list = []
                    for frame in frames:
                        f = get_formants(frame.unsqueeze(0), sample_rate)
                        formants_list.append(f)
                    formants_tensor = torch.tensor(np.array(formants_list).T, dtype=torch.float32)  # shape: (3, num_frames)
                    #interpolate to match MFCC time dimension
                    formants_tensor = torch.nn.functional.interpolate(
                        formants_tensor.unsqueeze(0), size=mel_spec.shape[-1], mode="linear", align_corners=False
                    )

                    formants_tensor = torch.log1p(formants_tensor)

                    pitch_diff = torch.abs(pitch[:, :, 1:] - pitch[:, :, :-1])
                    jitter = torch.mean(pitch_diff, dim=-1, keepdim=True)
                    jitter = torch.nn.functional.interpolate(jitter, size=mel_spec.shape[-1], mode="linear")

                    harmonic_energy = mag[:, :20].mean(dim=1, keepdim=True)
                    noise_energy = mag[:, 20:].mean(dim=1, keepdim=True)
                    hnr = harmonic_energy / (noise_energy + 1e-6)
                    hnr = torch.log1p(hnr)

                    voiced = (pitch > 0).float()
                    speech_rate = voiced.mean(dim=-1, keepdim=True)
                    speech_rate = torch.nn.functional.interpolate(speech_rate, size=mel_spec.shape[-1], mode="linear")
                    mel_spec = torch.cat([mel_spec, speech_rate], dim=1)


                    mel_spec = torch.cat([mel_spec, pitch], dim=1)
                    mel_spec = torch.cat([mel_spec, spectral_centroid], dim=1)
                    mel_spec = torch.cat([mel_spec, centroid.unsqueeze(0)], dim=1)
                    mel_spec = torch.cat([mel_spec, bandwidth.unsqueeze(0)], dim=1)
                    mel_spec = torch.cat([mel_spec, rolloff.unsqueeze(0)], dim=1)
                    mel_spec = torch.cat([mel_spec, flatness.unsqueeze(0)], dim=1)
                    mel_spec = torch.cat([mel_spec, jitter], dim=1)
                    mel_spec = torch.cat([mel_spec, hnr], dim=1)

                    
                    mel_spec = mel_spec.unsqueeze(0) # B, C, Bins, Time

                    with open(self.path + csv_text + name + "/" + row[7] + "/" + file_name[:-4], "xb") as r:
                        pickle.dump(mel_spec, r)

def make_mfcc_output(number):
    d = DatasetProcessing(PATH, number, multiprocessing.cpu_count())
    d.create_mel_c("test", 64, 40, "_data")
    d.create_mel_c("train", 64, 40, "_data")

if __name__ == "__main__":

    p = [multiprocessing.Process(target=make_mfcc_output, args=[i]) for i in range(multiprocessing.cpu_count())]

    print(f"Starting {multiprocessing.cpu_count()} processes")

    for ii, processes in enumerate(p):
        if(psutil.virtual_memory().available > 4*(1024)**3):
            processes.start()
            print(f"Process {ii} started")
            time.sleep(5)
        else:
            while(psutil.virtual_memory().available < 4*(1024)**3):
                time.sleep(1)
            processes.start()
            print(f"Process {ii} started")
            time.sleep(5)
        
    for ii, processes in enumerate(p):
        processes.join()
        print(f"Process {ii} finished")

