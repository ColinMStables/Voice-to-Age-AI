import torch
import torch.nn as nn

class Audio_Transformer(nn.Module):
    def __init__(self, 
                 d_model = 128,
                 n_mels = 64,
                 classes = 8,
                 nheads = 8,
                 N = 6
                 ) -> None:
        super().__init__()

        self.input_embedding = nn.Linear(n_mels, d_model)

        self.pos_embedding = nn.Parameter(torch.randn((1, 400, d_model)))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, N)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, classes)
        )

    def forward(self, mel_spec):
        """
        mel_spec : (channels, n_mels, time)
        """
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)


        x = mel_spec.transpose(1, 2)

        x = self.input_embedding(x)

        pos = self.pos_embedding[:, :x.size(1), :]
        x = x + pos

        x = self.encoder(x) 

        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits