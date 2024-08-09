import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import math

class FuseBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(FuseBlock, self).__init__()

        self.fuse0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.fuse00 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.fuse000 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.fuse11 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.fuse111 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x0, x00, x000 = self.fuse0(x), self.fuse00(x), self.fuse000(x)
        x = x + x0 + x00 + x000
        x1, x2, x3 = self.fuse1(x), self.fuse11(x), self.fuse111(x)
        x = x + x1 + x2 + x3
        return x

class arch1000_2(nn.Module):

    def __init__(self):
        super(arch1000_2, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
        )

        self.proj_fuse_high = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_fuse_low = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
        )

        self.proj_fuse = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
        )
        
        self.fuse1 = FuseBlock(128, 64)
        self.fuse2 = FuseBlock(64, 32)
        self.fuse3 = FuseBlock(32, 16)
        self.fuse4 = FuseBlock(16, 8)
        self.fuse5 = FuseBlock(8, 4)
        self.fuse6 = FuseBlock(4, 2)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.PReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.abs(x[1]).unsqueeze(0)
        x3 = torch.angle(x[0]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0) # 3x1000x1000

        x1 = self.proj_mag_high(x1)
        x2 = self.proj_mag_low(x2)
        x3 = self.proj_phase_high(x3)
        x4 = self.proj_phase_low(x4)

        x1, x2 = x1 * x3, x2 * x4
        x1, x2 = self.proj_fuse_high(x1), self.proj_fuse_low(x2)
        x = x1 * x2
        x = self.proj_fuse(x)
        x = self.fuse1(x)
        x = self.fuse2(x)
        x = self.fuse3(x)
        x = self.fuse4(x)
        x = self.fuse5(x)
        x = self.fuse6(x)

        return self.head(x.view(x.size(0), -1))