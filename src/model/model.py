import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader




# =============================
# MODEL
# =============================


class SmallCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
