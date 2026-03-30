from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence_model = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def get_output_lengths(self, input_widths: Tensor) -> Tensor:
        lengths = torch.div(input_widths, 2, rounding_mode="floor")
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        return torch.clamp(lengths, min=1)

    def forward(self, images: Tensor, input_widths: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.feature_extractor(images)
        features = features.squeeze(2).permute(2, 0, 1)
        recurrent, _ = self.sequence_model(features)
        logits = self.classifier(recurrent)
        log_probs = logits.log_softmax(dim=2)
        output_lengths = self.get_output_lengths(input_widths.to(logits.device))
        return log_probs, output_lengths
