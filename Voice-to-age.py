import NeuralNet
import RTVoiceData
import torchaudio as ta
import torch
import torch.nn as nn
import pickle
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import glob
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time
import sys

import math

import csv
import os

import pandas as pd

torch.set_default_device("cuda:0")

PATH = "./cv-corpus-23.0-2025-09-05/en/"
BATCH_SIZE = 1024
TEST_BATCH_SIZE = 32
EP = 1E-6
WD = 1E-3

BATCHES_PER_TEST = 20

n_mel = 64


class ProgressBar:
    """
    
    Shows the progress bar through the batches, shows loss, accuracy, time left and phase number
    phase number changes which parts of the model are training
    
    """
    def __init__(self, total_batches, bar_length=30):
        self.total_batches = total_batches
        self.bar_length = bar_length
        self.start_time = None
        self.accuracy = 0

    def update(self, batch_idx, loss, accuracy, phase_num):
        if self.start_time is None:
            self.start_time = time.time()

        if(accuracy == None):
            accuracy = self.accuracy
        else:
            self.accuracy = accuracy
        
        fraction_done = (batch_idx + 1) / self.total_batches
        filled_length = int(self.bar_length * fraction_done)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)

        # Estimates time left
        elapsed = time.time() - self.start_time
        est_total = elapsed / fraction_done if fraction_done > 0 else 0
        remaining = est_total - elapsed

        sys.stdout.write(
            f'\r[{bar}] {fraction_done*100:6.2f}% | Loss: {loss:.4f} | Acc: {accuracy:.2f}% | ETA: {remaining:6.1f}s | Batch Current {batch_idx} | Batch Total {self.total_batches} | Phase Number {phase_num}'
        )
        sys.stdout.flush()

        # If it's finished we move to the next line
        if batch_idx + 1 == self.total_batches:
            print()  

def collate_fn(batch):
    """
    Pads or truncates mel spectrograms because torch datasets need all to be the same size

    Args:
        batch: (mel, label)

    Returns:
        batch_mels: (B, M, T) tensor
        batch_labels: (B,) tensor
        lengths: (B,) original lengths
    """

    # Unziping the files seperates our spectrograms and labels
    mels, labels = zip(*batch)
    
    # All these calls prevent an error from torch
    mels = [m.detach().clone().float().cuda() if isinstance(m, torch.Tensor)
            else torch.tensor(m, dtype=torch.float32) for m in mels]
    
    # Gets the original lengths
    lengths = torch.tensor([mel.shape[-1] for mel in mels], dtype=torch.float32).cuda()

    # Takes the max length
    max_len = lengths.max().item()

    pad_len = int(max_len)

    padded_mels = []
    for mel in mels:
        # Pads if too short
        pad_amt = pad_len - mel.shape[-1]
        if pad_amt > 0:
            mel = F.pad(mel, (0, pad_amt))
        padded_mels.append(mel)

    batch_mels = torch.stack(padded_mels) # (B, M, pad_len)
    batch_labels = torch.tensor(labels, dtype=torch.long) # (B,)
    return batch_mels, batch_labels, lengths

class VoiceData(Dataset):
    """
    
    Uses pytorch datasets
    
    """

    def __init__(self,
                 root_dir
                 ) -> None:
        super().__init__()
        self.root_dir = root_dir

        self.root_dir = root_dir

        self.training = False

        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {"teens" : 0,
                             "twenties" : 1,
                             "thirties" : 2,
                             "fourties" : 3,
                             "fifties" : 4,
                             "sixties" : 5,
                             "seventies" : 6,
                             "eighties" : 7,
                             "nineties" : 8
                             }
        
        self.idx_to_class = ["teens", "twenties", "thrities", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]

        # We use glob to grab all of our spectrogram files
        self.files_by_class = {}
        for cls in class_names:
            cls_files = glob.glob(os.path.join(root_dir, cls, "*"))
            self.files_by_class[cls] = cls_files

        # Since the dataset is very skewed towards twenties we sample all the other classes until it matches its size
        if(self.training):
            max_count = max(len(files) for files in self.files_by_class.values())
            self.balanced_files = []
            for cls, files in self.files_by_class.items():
                sampled_files = random.choices(files, k=max_count)
                self.balanced_files.extend(sampled_files)

            random.shuffle(self.balanced_files)

            self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
            self.time_mask = T.TimeMasking(time_mask_param=20)
        else:
            self.balanced_files = self.files_by_class

    
    def __getitem__(self, idx):

        if(self.training):
            file_path = self.balanced_files[idx]
        else:
            file_path = self.files_by_class[self.idx_to_class[idx]]
        with open(file_path, "rb") as f:
            mel = pickle.load(f)

        # Avoids small values
        mel = torch.clamp(mel, min=1e-5)

        mel = mel.log()

        if self.training:
            mel = self.freq_mask(mel.cuda())
            mel = self.time_mask(mel.cuda())

        # This normalizes the spectrograms
        mean = mel.mean()
        std = mel.std()
        mel = (mel - mean) / (std + 1e-6)

        label_name = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[label_name]
        return mel, label

    def __len__(self):
        if(self.training):
            return len(self.balanced_files)
        else:
            return 0
    
"""

Instead of training everything at once, we train parts of the layers individually so it can learn better

Phase 1: triaining the classifier
Phase 2: training the cropping convolutional net
Phase 3: training both with a very small learning rate

"""

# These can freeze and unfreeze a layer
def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    module.train()
    for p in module.parameters():
        p.requires_grad = True

# The learning rate is different between phases
LR1 = 3e-4
LR2 = 1e-4
LR3 = 1e-5

def prepare_phase_1(model):
    freeze(model.second_finder)
    freeze(model.conv_net)        
    unfreeze(model.encoder)
    unfreeze(model.input_embedding)
    unfreeze(model.classifier)

def prepare_phase_2(model):
    unfreeze(model.second_finder)

    freeze(model.encoder)
    freeze(model.input_embedding)
    freeze(model.classifier)
    freeze(model.conv_net)

def prepare_phase_3(model):
    unfreeze(model.second_finder)
    unfreeze(model.encoder)
    unfreeze(model.input_embedding)
    unfreeze(model.conv_net)
    unfreeze(model.classifier)

# The optimizer has to be refreshed for the new parameters
def build_optimizer(model, phase):
    if phase == 1:
        # train classifier + encoder + conv_net
        params = []
        params += list(p for p in model.classifier.parameters() if p.requires_grad)
        params += list(p for p in model.encoder.parameters() if p.requires_grad)
        params += list(p for p in model.input_embedding.parameters() if p.requires_grad)
        params += list(p for p in model.conv_net.parameters() if p.requires_grad)

        return torch.optim.Adam(params, lr=LR1, eps = EP, weight_decay=WD)

    elif phase == 2:
        # train only second_finder
        params = list(p for p in model.second_finder.parameters() if p.requires_grad)
        return torch.optim.Adam(params, lr=LR2, eps = EP, weight_decay=WD)

    else:
        # train all modules together
        params = list(p for p in model.parameters() if p.requires_grad)
        return torch.optim.Adam(params, lr=LR3, eps = EP, weight_decay=WD)

def test_network(network, progress, max_accuracy, ii, loss, phase_num):
    """
    Tests the network

    Args:
        network: the classifier
        progress: the progress bar
        max_accuracy: the current best accuracy of the model
        ii: to display step number
        loss:  to display loss
        phase_num: to display phase number

    Returns:
        Max_accuracy: the new/old max accuracy
        Accuracy: the accuracy that was found from training

    """
    network.eval()
    test_batch_size = TEST_BATCH_SIZE
    test_dataset = VoiceData(PATH + "test")
    test_dataset.training = False
    test_dataloader = DataLoader(test_dataset, test_batch_size, False, generator=torch.Generator(device="cuda"), collate_fn= lambda x : collate_fn(x))
    correct = 0
    total = 0
    for batch in test_dataloader:
        
        output = network(batch[0], batch[2])

        _ , max_ind = output.max(dim=1)

        labels = batch[1]
        correct += (max_ind == labels).sum().item()

        total += labels.size(0)

    progress.update(ii, loss.item(), 100 * correct / total, phase_num)

    if(100* correct / total > max_accuracy):
        torch.save(network.state_dict(), "./Max_Accuracy_Model.pth")
        max_accuracy = 100* correct / total
    return max_accuracy, 100* correct / total

def count_parameters(network, only_trainable=True):
    """
    Returns the number of parameters in the network
    If only_trainable=True, counts only trainable parameters
    """
    if only_trainable:
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in network.parameters())
    
    
def save_training_log(file_path, loss, accuracy, step_increment=20):
    """
    Appends training statistics to a CSV file, if the csv is filled it starts from the last step

    Args:
        file_path: path to CSV file
        loss: loss of the model
        accuracy: accuracy of the model
        step_increment: how many steps passed (batches per training)
    """
    header = ["step", "loss", "accuracy"]
    # Check if file exists
    if os.path.isfile(file_path):
        # Read last step
        try:
            df = pd.read_csv(file_path)
            last_step = df['step'].iloc[-1] # Grabs the last line and checks step value
        except Exception:
            last_step = -step_increment  # In case the file fails
        step = last_step + step_increment
    else:
        step = 0  # start from 0

    # Append row
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([step, loss, accuracy])

def plot_training(log_file, ax1, ax2):
    """
    
    Plots the reward and accuracy of the model while it's training

    """
    if not os.path.isfile(log_file):
        return

    df = pd.read_csv(log_file)
    steps = df['step']
    loss = df['loss']
    accuracy = df['accuracy']

    ax1.clear()
    ax2.clear()

    # Loss (left side)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(steps, loss, 'r-', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    # Accuracy (right side)
    ax2.set_ylabel('Accuracy (%)', color='tab:blue', labelpad=15)
    ax2.yaxis.set_label_position("right")
    ax2.plot(steps, accuracy, 'b-', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# These are how long each phase should last
PHASE_1_Percentage = 0.3
PHASE_2_Percentage = 0.3
PHASE_3_Percentage = 0.4

dataset_length = None

def train(model : NeuralNet.Audio_Transformer, total_epochs=50, training_log_path = "./training_log.csv"):
    """
    
    Trains the model for a number of epochs and logs the data to a csv

    """

    max_accuracy = 0
    device = torch.device("cuda")

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    for epoch in range(total_epochs):

        dataset_length = None

        print(f"\n======== ({epoch} Epochs) ========\n")

        # Prepare model modules
        prepare_phase_1(model)

        phase_num = 1

        # Optimizer for this phase
        optimizer = build_optimizer(model, 1)

        dataset = VoiceData(PATH + "train")
        dataset.training = True
        dataloader = DataLoader(dataset, BATCH_SIZE, True,
                                generator=torch.Generator(device="cuda"),
                                collate_fn=lambda x: collate_fn(x))
        
        if(dataset_length == None):
            dataset_length = len(dataset)
            PHASE_1_Steps = int(PHASE_1_Percentage * (dataset_length // BATCH_SIZE))
            PHASE_2_Steps = int(PHASE_2_Percentage * (dataset_length // BATCH_SIZE))
            PHASE_2_not_init = True
            PHASE_3_not_init = True

        progress = ProgressBar(len(dataset) // BATCH_SIZE)

        for ii, batch in enumerate(dataloader):

            # If we've passed the number of steps to get to the next phase we freeze/unfreeze layers and refresh the optimizer
            if(ii > PHASE_1_Steps):
                if(PHASE_2_not_init):
                    PHASE_2_not_init = False
                    prepare_phase_2(model)
                    optimizer = build_optimizer(model, 2)
                    phase_num = 2

            if(ii > (PHASE_2_Steps + PHASE_1_Steps)):
                if(PHASE_3_not_init):
                    PHASE_3_not_init = False
                    prepare_phase_3(model)
                    optimizer = build_optimizer(model, 3)
                    phase_num = 3


            model.train()
            mel = batch[0].to(device)
            labels = batch[1].to(device).long()

            output = model(mel.to(device), batch[2])
            loss = nn.CrossEntropyLoss()(output, labels) # Gets the mel spectrogram and labels as the index numbers

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            progress.update(ii, loss.item(), None, phase_num)

            # Tests the model and updates accuracy
            if (ii + 1) % BATCHES_PER_TEST == 0:
                max_accuracy, accuracy = test_network(model, progress, max_accuracy, ii, loss, phase_num)
                save_training_log(training_log_path, loss.item(), accuracy)
                plot_training(training_log_path, ax1, ax2)
        
                

network = NeuralNet.Audio_Transformer(n_mels=n_mel, classes=9, d_model=128, nheads=8, N=6, frame_length=100)
network.load_state_dict(torch.load("./Max_Accuracy_Model.pth"))

print(f"Total number of parameters: {count_parameters(network, False)}")

voice_data = RTVoiceData.RTVoice(n_mel=n_mel)

training_log = "./training.csv"

train(network, 500, training_log)
