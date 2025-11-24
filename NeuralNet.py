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

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.AvgPool1d(2),
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
        x = self.conv(x)   # (B, 256, 1)
        x = self.fc(x)     # (B,2)
        x = torch.sigmoid(x)
        return torch.clamp(x, 1e-4, 1 - 1e-4)

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
            nn.AvgPool1d(2),
            nn.BatchNorm1d(256),
            nn.ReLU()
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
                 frame_length=100
                 ):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.frame_length = frame_length

        self.debug = False

        #self.second_finder = RL_Second_Finder(n_mel=n_mels)

        self.conv_net = Direct_Convolution(n_mels)

        # 256 for the convolution channels #TODO make variable
        self.input_embedding = nn.Linear(256, d_model)

        self.max_pos = frame_length + 1 
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_pos * 4, d_model))

        self.pre_norm = nn.LayerNorm(d_model)
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, batch_first=True, dim_feedforward=1024)
        # self.encoder = nn.TransformerEncoder(encoder_layer, N)

        self.encoder = ta.models.Conformer(d_model, nheads, 1024, N, 3, 0.1)

        # Here we use the CLS token like bert for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, classes)
        )

    def crop_mel_spec_dual(self, mel_spec, left_norm, right_norm, lengths=None):
        """
        Left and right cropping of padded mel spectrograms that preserves gradients and avoids cropping padded areas

        Args:
            mel_spec: (B, C, M, T) padded mel spectrogram batch
            left_norm, right_norm: (B,) floats in [0,1] crop start / end positions
            lengths: (B,) the mel lengths before padding

        Returns:
            Cropped mel spectrogram: (B, C, M, frame_length)
        """

        B, C, M, T = mel_spec.shape
        device = mel_spec.device
        frame_length = self.frame_length

        # The length of the mel spectrograms are padded for batching purposes 
        # But if we want to crop the spectrograms we don't want to crop into a padded portion so we pass the original lengths
        if lengths is None:
            lengths = torch.full((B,), T, device=device, dtype=torch.float32)
        else:
            lengths = lengths.to(device, dtype=torch.float32)

        # Lengths shouldn't be zero but just in case
        lengths = torch.clamp(lengths, min=1)

        # This will be the [0-1] scale of the unpadded mel_spec compared to the full spec
        scale = (lengths / T).view(B, 1)

        # These should already be in the range [0,1], but in case they aren't
        left = left_norm.clamp(0.0, 1.0)

        # If the right crop is less than the left we pad it to be a little extra
        right = torch.maximum(right_norm.clamp(0.0, 1.0), left + 1e-5)

        # linspace in [0,1] for output time axis with frame_length divisions
        lin_x = torch.linspace(0.0, 1.0, frame_length, device=device).view(1, frame_length)

        # Base crop range [left, right]
        crop_range = left.unsqueeze(1) + lin_x * (right - left).unsqueeze(1)  # (B,frame)

        # Adjust so that range never exceeds each sample’s real length
        # The main issue with this is that it adjusts the left side too but the model should learn around that when cropping
        crop_range = crop_range * scale  # (B, frame_length)

        #
        #   This gets everything ready for the grid sampling
        #

        # Convert to [-1,1] normalized coords for grid sample
        grid_x = crop_range * 2.0 - 1.0
        grid_x = grid_x.unsqueeze(1).expand(B, M, frame_length)  # (B,M,frame)

        # We count the bins as the Y axis which should stay the same after cropping
        # To keep them the same we specify that we want grid_sample to use all the bins
        lin_y = torch.linspace(-1.0, 1.0, M, device=device).view(1, M, 1)
        grid_y = lin_y.expand(B, M, frame_length)

        # Puts them together
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (B, M, frame, 2)

        # So grid sample should contain grid_x of a cropped spectrogram by specifying the locations [0.2, 0.21, 0.22 , ...... 0.79, 0.8] for 0.2 left and 0.8 right with full scale
        # And our grid_y should go from [-1, 1] in mel_bins number of steps to preserve it

        # Grid sample will then interpolate the original mel_spec to a new cropped one, while also preserving the gradients of the left and right
        cropped = F.grid_sample(
            mel_spec,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return cropped

    
    def plot_cropped_mel(self, mel_spec, cropped):
        """
        Visualizes:
        - The original mel spectrogram
        - The cropped spectrogram from crop_mel_spec_dual()
        
        model: the network containing crop_mel_spec_dual()
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

    def forward(self, mel_spec: torch.Tensor, lengths = None):
        """
        mel_spec: (B, 1, n_mels, time)
        returns logits: (B, classes)
        """
        B, C, M, T = mel_spec.shape
 
        x = mel_spec.squeeze(1)  # (B, M, T)

        # second = self.second_finder(x)  # (B,2)
        # start_norm, duration_norm = second[:,0], second[:,1]

        # # Cropping using the convolutional layer
        # cropped = self.crop_mel_spec_dual(mel_spec, start_norm, duration_norm, lengths=lengths)

        # if(self.debug):
        #     # This plots both the mel spectrogram before and after cropping
        #     self.plot_cropped_mel(sf_in, cropped)

        # #convert to (B, frame_length, n_mels)
        # x = cropped.squeeze(1)

        x = self.conv_net(x)

        x = x.transpose(1,2) # (B, frame_length, n_mels)

        # embedding
        x = self.input_embedding(x)  # (B, frame_length, d_model)

        # CLS token + pos embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,d)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, frame_length+1, d)

        pos = self.pos_embedding[:, :x.size(1), :].to(x.device)
        x = x + pos

        x = self.pre_norm(x)
        lens = torch.full((B,), x.size(1), dtype=torch.long, device=x.device)
        x = self.encoder(x, lens)
        #x = self.encoder(x)             # (B, length+1, d)
        cls_output = x[0][:, 0, :]         # (B, d)
        logits = self.classifier(cls_output)   # (B, classes)
        return logits
