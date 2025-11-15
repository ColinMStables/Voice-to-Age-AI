"""

Processes the mcv dataset into the mel spectrograms TODO Make this nicer

"""

import csv
import torchaudio as ta
import pickle
import os

PATH = "./cv-corpus-23.0-2025-09-05/en/"

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


d = DatasetProcessing(PATH)
