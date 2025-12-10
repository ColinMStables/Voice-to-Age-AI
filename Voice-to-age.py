import NeuralNet
import RTVoiceData
import torchaudio as ta
import torch
import torch.nn as nn
import pickle
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from collections import defaultdict
import torchaudio.transforms as T
import glob
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch.nn.functional as F
from DataProcessing import DatasetProcessing
import soundfile as sf

import time
import sys

import math

import csv
import os

import pandas as pd
import psutil
from torch.utils.data import Sampler

torch.set_default_device("cuda:0")

PATH = "./cv-corpus-23.0-2025-09-05/en/"
BATCH_SIZE = 128
TEST_BATCH_SIZE = 32
EP = 1E-6
WD = 1E-4

BATCHES_PER_TEST = 10

# There are 3 phases of training these define the three learning rates
LR1 = 1e-6
LR2 = 3e-4
LR3 = 1e-6

ONE_D = True

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

        # Estimates time left based on time from previous update call
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
    Pytorch dataset for testing and training models

    Args:
        root_dir : where the datset is, should be to the path provided by data_processing.py
        training : removes the RAM cache feature if this is false
        leave_free_gb: dataset is cached in RAM so this leaves a certain number of GB free from memory
    """

    def __init__(self,
                 root_dir,
                 training=True,
                 leave_free_gb=3
                 ):
        super().__init__()

        self.root_dir = root_dir
        self.training = training
        self.leave_free_gb = leave_free_gb

        if(os.path.isfile("./datacache")):
            with open("./datacache", "rb") as f:
                self.properties = pickle.load(f)
        else:
            self.properties = {
            "processed" : False,
            "total_file_size" : None
            }
            open("./datacache", "xb")


        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        
        # Selects all the files inside the dataset folder
        self.files = []
        for cls in class_names:
            self.files.extend(glob.glob(os.path.join(root_dir, cls, "*")))

        random.shuffle(self.files)

        #
        # Cache system
        #
        if(training):

            if(self.properties["processed"] == False):
                print("removing data")
                for f in self.files:
                    try:
                        with open(f, "rb") as x:
                            pickle.load(x)
                    except KeyboardInterrupt as e:
                        print(f"ending processing")
                        break
                    except BaseException as e:
                        os.remove(f)
                        print(f"removed {f} for reason {e}")
                        self.files.remove(f)
                self.properties["processed"] = True
                print("finished processing data")

            if(self.properties["total_file_size"] !=None):
                file_sizes = self.properties["total_file_size"]
            else:
                file_sizes = [os.path.getsize(f) for f in self.files]
                self.properties["total_file_size"] = file_sizes

            vm = psutil.virtual_memory()
            free_ram = vm.available
            reserve_bytes = self.leave_free_gb * (1024**3)

            if free_ram <= reserve_bytes:
                cache_budget = 0
            else:
                cache_budget = free_ram - reserve_bytes

            print(f"Free RAM: {free_ram/1e9:.2f} GB")
            print(f"Reserving: {reserve_bytes/1e9:.2f} GB")
            print(f"Cache budget: {cache_budget/1e9:.2f} GB")

            cached_paths = []
            used_bytes = 0

            for path, size in zip(self.files, file_sizes):
                if used_bytes + size > cache_budget:
                    break
                cached_paths.append(path)
                used_bytes += size

            self.cached_paths = set(cached_paths)
            self.disk_paths = set(self.files) - self.cached_paths

            print(f"Caching {len(self.cached_paths)} files "
                f"({used_bytes/1e9:.2f} GB used).")
            print(f"{len(self.disk_paths)} files stay on disk.")

            self.ram_cache = {}
            for path in self.cached_paths:
                with open(path, "rb") as f:
                    self.ram_cache[path] = pickle.load(f).cpu()
            
            with open("./datacache", "wb") as f:
                pickle.dump(self.properties, f)


    def load_mel(self, path):
        """
        Checks to see if this is cached, if not load it into memory
        """
        if not self.training:
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            if path in self.ram_cache:
                return self.ram_cache[path].clone()
            else:
                
                with open(path, "rb") as f:
                    return pickle.load(f)
        except:
            with open(path, "rb") as f:
                return pickle.load(f)


    
    def __getitem__(self, idx):
        file_path = self.files[idx]

        mel = self.load_mel(file_path)

        # Avoids small values
        mel = torch.clamp(mel, min=1e-5)

        #mel = mel.log()

        # This normalizes the spectrograms (Removed)
        # mean = mel.mean()
        # std = mel.std()
        # mel = (mel - mean) / (std + 1e-6)

        label_name = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[label_name]
        return mel, label


    def __len__(self):
        return len(self.files)

class DynamicWeightedSampler(Sampler):
    def __init__(self, dataset, class_accuracy=None):
        self.dataset = dataset
        self.class_to_idx = dataset.class_to_idx
        self.files = dataset.files

        self.class_accuracy = (
            class_accuracy or {cls_idx: 50 for cls_idx in self.class_to_idx.values()}
        )

        self.update_weights()

    def update_weights(self, class_accuracy=None):
        if class_accuracy is not None:
            self.class_accuracy = class_accuracy

        weights = []
        for path in self.files:
            cls_name = os.path.basename(os.path.dirname(path))
            acc = self.class_to_idx[cls_name]
            weight = max(1.0, 100 - acc)
            weights.append(weight)

        self.weights = torch.tensor(weights, dtype=torch.float)

    def __iter__(self):
        return iter(
            torch.multinomial(
                self.weights,
                num_samples=len(self.weights),
                replacement=True
            ).tolist()
        )

    def __len__(self):
        return len(self.dataset)


class BalancedBatchSampler(Sampler):
    """
    Samples batches such that each batch contains roughly equal numbers of samples per class.
    Handles leftover samples by random selection.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = len(dataset.class_to_idx)
        assert batch_size >= self.num_classes, "Batch size must be >= number of classes"

        self.class_to_indices = {}
        for idx, path in enumerate(dataset.files):
            cls_name = os.path.basename(os.path.dirname(path))
            cls_idx = dataset.class_to_idx[cls_name]
            self.class_to_indices.setdefault(cls_idx, []).append(idx)

        self.classes = list(self.class_to_indices.keys())

        # Shuffle indices for each class
        for cls in self.classes:
            random.shuffle(self.class_to_indices[cls])

        # Compute samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        self.extra_samples = batch_size % self.num_classes  # fill remainder randomly

        # Compute number of batches per epoch
        self.max_class_len = max(len(idxs) for idxs in self.class_to_indices.values())
        self.num_batches = math.ceil(self.max_class_len / self.samples_per_class)

    def __iter__(self):
        class_pointers = {cls: 0 for cls in self.classes}

        for _ in range(self.num_batches):
            batch = []

            for cls in self.classes:
                idxs = self.class_to_indices[cls]
                start = class_pointers[cls]
                end = start + self.samples_per_class
                batch.extend(idxs[start:end])

                class_pointers[cls] = end
                if class_pointers[cls] >= len(idxs):
                    random.shuffle(idxs)
                    class_pointers[cls] = 0

            all_indices = [i for idxs in self.class_to_indices.values() for i in idxs]
            if self.extra_samples > 0:
                batch.extend(random.choices(all_indices, k=self.extra_samples))

            random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return self.num_batches


def make_weighted_sampler(dataset):

    # Count by class
    counts = {cls: 0 for cls in dataset.class_to_idx}
    for path in dataset.files:
        cls = os.path.basename(os.path.dirname(path))
        counts[cls] += 1
    print(counts)
    # Inverse frequency
    class_weights = {
        cls: 1.0 / count if count > 0 else 0.0
        for cls, count in counts.items()
    }
    # Per-sample weights
    sample_weights = []
    for path in dataset.files:
        cls = os.path.basename(os.path.dirname(path))
        sample_weights.append(class_weights[cls])
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sample_weights = torch.sqrt(sample_weights)
    sample_weights = sample_weights / sample_weights.sum()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # same number per epoch
        replacement=True
    )
    return sampler

def make_accuracy_weighted_sampler(dataset, class_accuracy):

    class_weights = {}
    for class_name, idx in dataset.class_to_idx.items():
        acc = class_accuracy.get(idx, None)

        if acc is None:
            # If missing: treat it as low-accuracy to encourage more sampling
            weight = 100.0
        else:
            # Prevent division by zero of extremely accurate classes
            weight = max(1.0, 100.0 - acc)

        class_weights[class_name] = weight

    sample_weights = []
    for path in dataset.files:
        class_name = os.path.basename(os.path.dirname(path))
        sample_weights.append(class_weights[class_name])

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # full epoch
        replacement=True
    )

    return sampler
    
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

def prepare_phase_1(model):
    freeze(model.second_finder)
    if(not ONE_D):
        unfreeze(model.conv_net)        
    unfreeze(model.encoder)
    unfreeze(model.input_embedding)
    unfreeze(model.classifier)

def prepare_phase_2(model):
    unfreeze(model.second_finder)

    freeze(model.encoder)
    freeze(model.input_embedding)
    freeze(model.classifier)
    if(not ONE_D):
        freeze(model.conv_net)

def prepare_phase_3(model):
    unfreeze(model.second_finder)
    unfreeze(model.encoder)
    unfreeze(model.input_embedding)
    if(not ONE_D):
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
        if(not ONE_D):
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
    
def save_test_log(
    file_path,
    loss,
    accuracy,
    top3_accuracy,
    class_accuracy: dict,
    step_increment=BATCHES_PER_TEST
):
    """
    Appends test statistics to a CSV file. Auto-increases 'step' just like save_training_log().
    
    Columns saved:
        step, loss, accuracy, top3_accuracy, class_0, class_1, ...

    Args:
        file_path: CSV path
        loss: average test loss
        accuracy: total top-1 accuracy
        top3_accuracy: total top-3 accuracy
        class_accuracy: dict mapping class_index -> class accuracy %
        step_increment: how many steps to advance
    """


    # base columns
    header = ["step", "loss", "accuracy", "top3_accuracy"]

    # add one column per class dynamically:
    max_class = max(class_accuracy.keys()) if class_accuracy else -1
    header += [f"class_{i}" for i in range(max_class + 1)]


    if os.path.isfile(file_path):
        try:
            df = pd.read_csv(file_path)
            last_step = df['step'].iloc[-1]
        except Exception:
            last_step = -step_increment
        step = last_step + step_increment
    else:
        step = 0


    row = [step, loss, accuracy, top3_accuracy]

    # add class accuracies in consistent order
    for i in range(max_class + 1):
        row.append(class_accuracy.get(i, None))  # None if class missing

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def test_network(network, progress, max_accuracy, ii, loss, phase_num, training_path):

    network.eval()
    test_batch_size = TEST_BATCH_SIZE
    test_dataset = VoiceData(PATH + "test_data", False)
    test_dataset.training = False
    test_dataloader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        generator=torch.Generator(device="cuda"),
        collate_fn=lambda x: collate_fn(x),
    )

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    # Track per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Optional: track loss across entire test set
    total_loss = 0.0
    loss_count = 0

    with torch.no_grad():
        for batch in test_dataloader:

            inputs, labels, cond = batch
            logits = network(inputs, cond)

            batch_loss = F.cross_entropy(logits, labels, reduction="mean")
            total_loss += batch_loss.item()
            loss_count += 1

            _, pred_top1 = logits.max(dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()

            top3 = torch.topk(logits, k=3, dim=1).indices
            correct_top3 += sum([labels[i] in top3[i] for i in range(labels.size(0))])

            for label, pred in zip(labels, pred_top1):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1

            total += labels.size(0)

    # Compute metrics
    accuracy = 100 * correct_top1 / total
    top3_accuracy = 100 * correct_top3 / total
    avg_loss = total_loss / loss_count

    # Per-class accuracy dictionary (percentage)
    class_accuracy = {
        cls: 100 * class_correct[cls] / class_total[cls]
        for cls in class_total
    }

    # Update progress bar
    progress.update(ii, loss.item(), accuracy, phase_num)

    # Save always
    torch.save(network.state_dict(), "./Current_Model.pth")

    # Save best
    if accuracy > max_accuracy:
        torch.save(network.state_dict(), "./Max_Accuracy_Model.pth")
        max_accuracy = accuracy

    save_test_log(
        training_path,
        loss=avg_loss,
        accuracy=accuracy,
        top3_accuracy=top3_accuracy,
        class_accuracy=class_accuracy
    )

    return max_accuracy, accuracy, top3_accuracy, class_accuracy


def count_parameters(network, only_trainable=True):
    """
    Returns the number of parameters in the network
    If only_trainable=True, counts only trainable parameters
    """
    if only_trainable:
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in network.parameters())
    
def build_dataloader(dataset, sampler):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        generator=torch.Generator(device="cuda"),
        collate_fn=lambda x: collate_fn(x)
    )



def plot_all_metrics(log_file, ax_left, ax_right):
    """
    Plots:
        - Loss (left Y-axis, RED)
        - Accuracy (right Y-axis)
        - Top-3 Accuracy (right Y-axis)
        - Per-class Accuracy curves (right Y-axis)
    """

    if not os.path.isfile(log_file):
        return

    df = pd.read_csv(log_file)

    # Required
    steps = df["step"]
    loss = df["loss"]
    accuracy = df["accuracy"]

    # Optional
    has_top3 = "top3_accuracy" in df.columns
    per_class_cols = [c for c in df.columns if c.startswith("class_")]

    # Clear axes
    ax_left.clear()
    ax_right.clear()


    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Loss", color="red")
    ax_left.plot(steps, loss, color="red", label="Loss", linewidth=2)
    ax_left.tick_params(axis='y', labelcolor="red")


    ax_right.set_ylabel("Accuracy (%)", labelpad=15)
    ax_right.yaxis.set_label_position("right")

    # Main accuracy
    ax_right.plot(steps, accuracy, label="Accuracy", linewidth=2)

    # Top-3 accuracy
    if has_top3:
        ax_right.plot(steps, df["top3_accuracy"], label="Top-3 Accuracy", linestyle="--", linewidth=2)

    root_dir = PATH + "test_data"
    class_names = sorted(os.listdir(root_dir))

    col_to_class = {}
    for col in per_class_cols:
        idx = int(col.replace("class_", ""))
        if idx < len(class_names):
            col_to_class[col] = class_names[idx]
        else:
            col_to_class[col] = f"Class {idx}"

    # Per-class accuracy curves
    for col in per_class_cols:
        label_name = col_to_class[col]
        ax_right.plot(steps, df[col], label=label_name, alpha=0.3)


    # ──────────────────────────────
    # Legends
    # ──────────────────────────────
    ax_left.legend(loc="upper left")
    ax_right.legend(loc="upper right")

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)



# These are how long each phase should last
PHASE_1_Percentage = 0.2
PHASE_2_Percentage = 0.2
PHASE_3_Percentage = 0.6

dataset_length = None

def train_batch_sampler(model : NeuralNet.Audio_Transformer, total_epochs=50, training_log_path = "./training_log.csv"):
    """
    
    Trains the model for a number of epochs and logs the data to a csv

    """

    max_accuracy = 0
    device = torch.device("cuda")

    print("Start in debug mode? y/n")
    ans = input()

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    training = True
    
    if(ans == "y"):
        training = False
        model.debug = True
        print("Starting in debug mode")

    dataset = VoiceData(PATH + "train_data", training, leave_free_gb=3)

    sampler = BalancedBatchSampler(dataset, BATCH_SIZE)

    for epoch in range(total_epochs):

        dataset_length = None

        print(f"\n======== ({epoch} Epochs) ========\n")

        # Prepare model modules
        prepare_phase_1(model)

        phase_num = 1

        # Optimizer for this phase
        optimizer = build_optimizer(model, 1)

        dataset.training = True

        dataloader = DataLoader(dataset, BATCH_SIZE, sampler=sampler,
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

            #If we've passed the number of steps to get to the next phase we freeze/unfreeze layers and refresh the optimizer
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

            #print(torch.bincount(labels, minlength=9))

            output = model(mel.to(device), batch[2])

            # === LOGIT SAFETY CHECK ===
            if not torch.isfinite(output).all():
                continue

            # === LABEL SAFETY CHECK ===
            labels = labels.long()
            if labels.min() < 0 or labels.max() >= output.shape[1]:
                continue

            loss = nn.CrossEntropyLoss(label_smoothing = 0.1)(output, labels)

            # === LOSS SAFETY CHECK ===
            if not torch.isfinite(loss):
                continue

            loss.backward()

            # === GRADIENT SAFETY CHECK ===
            bad_grad = False
            for n, p in network.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    bad_grad = True

            if bad_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()  

            progress.update(ii, loss.item(), None, phase_num)

            # Tests the model and updates accuracy
            if (ii + 1) % BATCHES_PER_TEST == 0:
                test_network(model, progress, max_accuracy, ii, loss, phase_num, training_log_path)
                plot_all_metrics(training_log_path, ax1, ax2)

def train(model : NeuralNet.Audio_Transformer, total_epochs=50, training_log_path = "./training_log.csv"):
    """
    
    Trains the model for a number of epochs and logs the data to a csv

    """

    max_accuracy = 0
    device = torch.device("cuda")

    print("Start in debug mode? y/n")
    ans = input()

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    training = True
    
    if(ans == "y"):
        training = False
        model.debug = True
        print("Starting in debug mode")

    dataset = VoiceData(PATH + "train_data", training, leave_free_gb=3)

    sampler = DynamicWeightedSampler(dataset, BATCH_SIZE)

    for epoch in range(total_epochs):

        dataset_length = None

        print(f"\n======== ({epoch} Epochs) ========\n")

        # Prepare model modules
        prepare_phase_1(model)

        phase_num = 1

        # Optimizer for this phase
        optimizer = build_optimizer(model, 1)

        dataset.training = True

        dataloader = DataLoader(dataset, BATCH_SIZE, sampler=sampler,
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

            print(torch.bincount(labels, minlength=9))

            output = model(mel, batch[2])

            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

            progress.update(ii, loss.item(), None, phase_num)

            # --- TEST & REBUILD SAMPLER / LOADER ---
            if (ii + 1) % BATCHES_PER_TEST == 0:

                _, _, _, class_acc = test_network(model, progress, max_accuracy, ii, loss, phase_num, training_log_path)
                plot_all_metrics(training_log_path, ax1, ax2)

                sampler.update_weights(class_acc)

                break  # leave the for-loop so outer loop will recreate iteration

        
                

network = NeuralNet.Audio_Transformer(n_mels=49, classes=9, d_model=128, nheads=16, N=6, frame_length=300, classifier_d=256, one_d=ONE_D)
#network.load_state_dict(torch.load("./Current_Model.pth"))

print(f"Total number of parameters: {count_parameters(network, False)}")
print(f"Transformer number of parameters: {count_parameters(network.encoder, False)}")
if(not ONE_D):
    print(f"Feed in convolution number of parameters: {count_parameters(network.conv_net, False)}")
print(f"Convolution second finder of parameters: {count_parameters(network.second_finder, False)}")

voice_data = RTVoiceData.RTVoice(n_mel=n_mel)

training_log = "./training.csv"

# torch.autograd.set_detect_anomaly(True)

train(network, 5000000, training_log)
