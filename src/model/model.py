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

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.network(x)  # (B, 1, H, W)
        # back to (B, H, W)
        return x.squeeze(1)



    
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
        # x: (B, H, W, C)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.network(x)  # (B, 1, H, W)
        return x.squeeze(1)