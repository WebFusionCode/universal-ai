import torch
import torch.nn as nn


class TabTransformer(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.embedding = nn.Linear(input_dim, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.fc(x)