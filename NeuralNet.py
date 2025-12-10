import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

import matplotlib.pyplot as plt

class RL_Second_Finder(nn.Module):
    """
    Predicts where to crop the left and right sides of a mel spectrogram

    To run on an embedded system this needs to be very small but accurate
    """
    
    def __init__(self, n_mel=64, hidden=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_mel, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid()                 
        )

    def forward(self, x):
        # x: (B, n_mels, time)
        h = self.conv(x).squeeze(-1)   # (B, 256)
        out = self.fc(h)               # (B, 2)  raw a, b

        # return (start, right) as shape (B,2)
        return out
    
class RL_Second_Finder_2D(nn.Module):
    """
    Predicts where to crop the left and right sides of a mel spectrogram

    To run on an embedded system this needs to be very small but accurate
    """
    
    def __init__(self, n_mel=64, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_mel, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)               
        )

    def forward(self, x):
        # x: (B, n_mels, time)
        h = self.conv(x).squeeze(-1)   # (B, 256)
        out = self.fc(h)               # (B, 2)  raw a, b

        # return (start, right) as shape (B,2)
        return out

class Direct_Convolution(nn.Module):

    def __init__(self,
                 n_mel = 64
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_mel, 128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(2)
         )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Direct_Convolution_2D(nn.Module):

    def __init__(self,
                 n_mel = 64,
                 d_out = 200
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_mel, 128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.AdaptiveAvgPool2d((40, d_out)),
         )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class Audio_Transformer(nn.Module):
    def __init__(self, 
                 d_model=128,
                 n_mels=64,
                 classes=9,
                 nheads=8,
                 N=6,
                 frame_length=100,
                 classifier_d = 1024,
                 one_d = True
                 ):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.frame_length = frame_length
        self.one_d = one_d
        self.debug = False

        if(one_d):
            self.second_finder = RL_Second_Finder(n_mel=n_mels)

            self.input_embedding = nn.Linear(n_mels, d_model)
        else:
            self.second_finder = RL_Second_Finder_2D(n_mel=1)

            self.conv_net = Direct_Convolution_2D(1)

            self.input_embedding = nn.Linear(128, d_model)


        self.max_pos = frame_length + 1 
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_pos * 4, d_model))

        self.pre_norm = nn.LayerNorm(d_model)
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, batch_first=True, dim_feedforward=1024)
        # self.encoder = nn.TransformerEncoder(encoder_layer, N)

        self.encoder = ta.models.Conformer(d_model, nheads, 256, N, 5, 0.1, True, True)

        # Here we use the CLS token like bert for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_d, classifier_d //2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_d // 2, classes)
        )

    def crop_mel_spec_dual(self, mel_spec, left_norm, right_norm, lengths=None):
        # mel_spec: (B, C, M, T)
        B, C, M, T = mel_spec.shape
        device = mel_spec.device
        frame_length = self.frame_length

        if lengths is None:
            lengths = torch.full((B,), float(T), device=device)
        else:
            lengths = lengths.to(device).float()
        lengths = torch.clamp(lengths, min=1.0)

        # ensure left_norm/right_norm are floats
        left = left_norm.view(B, 1).to(device).float().clamp(0.0, 1.0)
        right = right_norm.view(B, 1).to(device).float().clamp(0.0, 1.0)

        # scale to real (avoid division by zero)
        scale = (lengths / max(float(T), 1.0))
        scale = scale.view(B, 1).clamp(0.0, 1.0)

        lin_x = torch.linspace(0.0, 1.0, frame_length, device=device).view(1, frame_length)
        crop_range = left + lin_x * (right - left)   # (B, frame_length)
        crop_range = crop_range * scale

        grid_x = crop_range * 2.0 - 1.0
        grid_x = grid_x.unsqueeze(1).expand(B, M, frame_length)   # (B, M, frame_length)

        lin_y = torch.linspace(-1.0, 1.0, M, device=device).view(1, M, 1)
        grid_y = lin_y.expand(B, M, frame_length)

        grid = torch.stack((grid_x, grid_y), dim=-1)  # (B, M, frame_length, 2)

        cropped = F.grid_sample(
            mel_spec,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        return cropped


    
    def plot_cropped_mel(self, mel_spec, cropped):
        """
        Visualizes:
        - The original mel spectrogram
        - The cropped spectrogram from crop_mel_spec_dual()
        
        mel_spec: tensor of shape (B, M, T)
        left_norm, right_norm: floats in [0,1]
        """


        # Pick first element and first channel
        original_img = mel_spec[0].clone().detach().cpu().numpy()
        cropped_img = cropped[0, 0].clone().detach().cpu().numpy()

        # Plot
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.title(f"Original Mel-Spectrogram\nShape: {original_img.shape}")
        plt.imshow(original_img, aspect='auto', origin='lower')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f"Cropped Mel-Spectrogram\nShape: {cropped_img.shape}")
        plt.imshow(cropped_img, aspect='auto', origin='lower')
        plt.colorbar()

        plt.tight_layout()
        plt.ioff()
        plt.show()

    def forward_1D(self, mel_spec: torch.Tensor, lengths = None):
        # mel_spec expected: (B, 1, 1, n_mels, time)

        mel_spec = mel_spec.squeeze(1)
        mel_spec[:,:,42:47,:] = torch.log1p(mel_spec[:, :, 42:47, :])
        mel_spec = torch.log1p(mel_spec)
        mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)

        assert mel_spec.dim() == 4, f"mel_spec dim={mel_spec.dim()}"
        B = mel_spec.size(0)
        # convert to (B, n_mels, time) for the second finder and conv1d
        sf_in = mel_spec.squeeze(1)   # (B, n_mels, time)

        # get crop coords (B,2)
        second = self.second_finder(sf_in)   # (B,2)
        start_norm, right_norm = second[:, 0], second[:, 1]

        # crop (works with original mel_spec shape)
        cropped = self.crop_mel_spec_dual(mel_spec, start_norm, right_norm, lengths=lengths)  # (B, C, M, frame_length)

        # optionally debug
        # if self.debug:
        #     self.plot_cropped_mel(sf_in, cropped)

        x = cropped.squeeze(1)  # (B, M, frame_length)

        x = x.transpose(1, 2)   # (B, time, channels)

        x = self.input_embedding(x)  # (B, time, d_model)

        # cls token and pos
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos = self.pos_embedding[:, :x.size(1), :].to(x.device)
        x = x + pos

        x = self.pre_norm(x)
        lens = torch.full((B,), x.size(1), dtype=torch.long, device=x.device)
        x = self.encoder(x, lens)

        # pool
        pooled = x[0][:, 0, :]   # (B, d_model) since CLS will be at index 0
        logits = self.classifier(pooled)
        return logits

    def forward_2D(self, mel_spec: torch.Tensor, lengths = None):
        # mel_spec expected: (B, 1, 1, n_mels, time)

        mel_spec = mel_spec.squeeze(1)
        mel_spec[:,:,42:47,:] = torch.log1p(mel_spec[:, :, 42:47, :])
        mel_spec = torch.log1p(mel_spec)
        mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)

        assert mel_spec.dim() == 4, f"mel_spec dim={mel_spec.dim()}"
        B = mel_spec.size(0)
        # convert to (B, n_mels, time) for the second finder and conv1d
        sf_in = mel_spec # (B, 1, n_mels, time)

        # get crop coords (B,2)
        second = self.second_finder(sf_in)   # (B,2)
        start_norm, right_norm = second[:, 0], second[:, 1]

        # crop (works with original mel_spec shape)
        cropped = self.crop_mel_spec_dual(mel_spec, start_norm, right_norm, lengths=lengths)  # (B, C, M, frame_length)

        # optionally debug
        # if self.debug:
        #     self.plot_cropped_mel(sf_in, cropped)

        x = self.conv_net(cropped)
        x = x.flatten(2)
        x = x.transpose(1, 2) 

        x = self.input_embedding(x)  # (B, time', d_model)

        # cls token and pos
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos = self.pos_embedding[:, :x.size(1), :].to(x.device)
        x = x + pos

        x = self.pre_norm(x)
        lens = torch.full((B,), x.size(1), dtype=torch.long, device=x.device)
        x = self.encoder(x, lens)

        # pool
        pooled = x[0][:, 0, :]   # (B, d_model) since CLS will be at index 0
        logits = self.classifier(pooled)
        return logits

    def forward(self, mel_spec: torch.Tensor, lengths=None):
        if(self.one_d):
            return self.forward_1D(mel_spec, lengths)
        else:
            return self.forward_2D(mel_spec, lengths)

