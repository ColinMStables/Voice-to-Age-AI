"""

Processes the mcv dataset into the mel spectrograms TODO Make this nicer

"""

import csv
import torchaudio as ta
import torch
import pickle
import os

PATH = "./cv-corpus-23.0-2025-09-05/en/"
print(torch.cuda.is_available())
if(torch.cuda.is_available()):
    torch.set_default_device("cuda:0")

class DatasetProcessing():

    def __init__(self,
                 path) -> None:
        self.path = path


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
            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            tsv_reader.__next__()
            
            for row in tsv_reader:
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
        
        with open(self.path + csv_text + ".tsv", "r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            tsv_reader.__next__()
            
            for row in tsv_reader:
                if(row[7] != ""):
                    if(not os.path.isdir(self.path  + csv_text + name + "/" + row[7])):
                        os.mkdir(self.path + csv_text + name + "/" + row[7])
                    file_name = row[1]
                    
                    if(os.path.isfile(self.path + csv_text + name + "/" + row[7] + "/" + file_name[:-4])):
                        continue

                    waveform, sample_rate = ta.load(self.path + "clips/" + file_name)


                    waveform = waveform.cuda()

                    mel_transform = ta.transforms.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc = n_mfcc,
                        log_mels=True,
                        melkwargs={
                        "n_fft" : 512,
                        "hop_length" : 512,
                        "n_mels" : n_mels,
                        }
                    ).to("cuda:0")

                    mel_spec = mel_transform(waveform)

                    pitch = ta.functional.detect_pitch_frequency(waveform, sample_rate)
                    pitch = pitch.unsqueeze(0)
                    pitch = torch.nn.functional.interpolate(pitch, size=mel_spec.shape[-1], mode="linear", align_corners=False)

                    pitch = torch.log1p(pitch)

                    stft = torch.stft(
                        waveform,
                        n_fft=10,
                        hop_length=512,
                        win_length=10,
                        window=torch.hann_window(10),
                        return_complex=True
                    )

                    magnitude = stft.abs() 

                    mel_spec = torch.cat([mel_spec, pitch], dim=1)
                    mel_spec = torch.cat([mel_spec, magnitude], dim = 1)

                    
                    with open(self.path + csv_text + name + "/" + row[7] + "/" + file_name[:-4], "xb") as r:
                        pickle.dump(mel_spec, r)


d = DatasetProcessing(PATH)

# waveform, sample_rate = ta.load(PATH + "clips/" + "common_voice_en_1.mp3")
# pitch = ta.functional.detect_pitch_frequency(waveform, sample_rate)
# print(sample_rate)
# mel_transform = ta.transforms.MFCC(
#     sample_rate=sample_rate,
#     n_mfcc = 52,
#     log_mels=True,
#     melkwargs={
#     "n_fft" : 1024,
#     "hop_length" : 512,
#     "n_mels" : 64
#     }
# )
# mel_spec = mel_transform(waveform)
# pitch = pitch.unsqueeze(0)
# pitch = torch.nn.functional.interpolate(pitch, size=mel_spec.shape[-1], mode="linear", align_corners=False)

# stft = torch.stft(
#     waveform,
#     n_fft=10,
#     hop_length=512,
#     win_length=10,
#     window=torch.hann_window(10),
#     return_complex=True
# )

# magnitude = stft.abs() 

# print(magnitude.shape)
# print(mel_spec.shape)
# print(pitch.shape)
# pitch = torch.log1p(pitch)
# print(pitch)
# print(mel_spec)
# magnitude = torch.log1p(magnitude)
# print(magnitude)

#d.create_mel_transforms("test",64)
d.create_mel_c("train", 64, 52, "_data")

