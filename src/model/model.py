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
        super(SmallCNN, self).__init__()
        # Using a simple encoder-decoder or bottleneck structure 
        # to process the embeddings into a single biomass value per pixel.
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, kernel_size=1) # Per-pixel regression
        )

    def forward(self, x):
        return self.network(x).squeeze(1) # Output shape: [Batch, H, W]



    


class PointWiseModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # safety against corrupted inputs
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.network(x).squeeze(1)  # (B, H, W)