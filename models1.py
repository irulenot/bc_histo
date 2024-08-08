import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import math

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.MaxPool2d()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class arch1000_6(nn.Module):

    def __init__(self):
        super(arch1000_6, self).__init__()

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
        
        self.fuse1 = BasicBlock(128, 64)
        # self.fuse11 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(128),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        # self.fuse111 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(128),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        self.fuse2 = BasicBlock(64, 32)
        # self.fuse22 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        # )
        # self.fuse222 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        # )
        self.fuse3 = BasicBlock(32, 16)
        # self.fuse33 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        # )
        # self.fuse333 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        # )
        self.fuse4 = BasicBlock(16, 8)
        # self.fuse44 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(8),
        #     nn.PReLU(),
        # )
        # self.fuse444 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(8),
        #     nn.PReLU(),
        # )
        self.fuse5 = BasicBlock(8, 4)
        # self.fuse55 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(8),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        # )
        # self.fuse555 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(8),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        # )
        self.fuse6 = BasicBlock(4, 2)
        # self.fuse66 = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(2),
        #     nn.PReLU(),
        # )
        # self.fuse666 = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(2),
        #     nn.PReLU(),
        # )

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
        x = self.fuse1(x) # + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) # + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) # + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) # + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) # + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) # + self.fuse66(x) + self.fuse666(x)

        return self.head(x.view(x.size(0), -1))