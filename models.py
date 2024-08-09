import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import math

# Bigger
class arch1000_01(nn.Module):

    def __init__(self):
        super(arch1000_01, self).__init__()

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
        
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse111 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse1111 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse2222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse3333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse4444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse5555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse6666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )

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
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x) + self.fuse1111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x) + self.fuse2222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x) + self.fuse3333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x) + self.fuse4444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x) + self.fuse5555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x) + self.fuse6666(x)

        return self.head(x.view(x.size(0), -1))

# Dropout
class arch1000_02(nn.Module):

    def __init__(self):
        super(arch1000_02, self).__init__()

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
        
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse111 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.drop = nn.Dropout(0.2)

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
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.drop(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.drop(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.drop(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.drop(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.drop(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)
        x = self.drop(x)

        return self.head(x.view(x.size(0), -1))

# Gray
class arch1000_04(nn.Module):

    def __init__(self):
        super(arch1000_04, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
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
        
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse111 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )

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
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)

        return self.head(x.view(x.size(0), -1))

####################

class arch3000_1(nn.Module):

    def __init__(self):
        super(arch3000_1, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )

        self.proj_fuse_high = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_fuse_low = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
        )

        self.proj_fuse = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
        )
        
        # self.fuse1 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        # self.fuse11 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        # self.fuse111 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse7 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse77 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse777 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse8 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse88 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse888 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(576, 256),
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
        # x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)
        x = self.fuse7(x) + self.fuse77(x) + self.fuse777(x)
        x = self.fuse8(x) + self.fuse88(x) + self.fuse888(x)

        return self.head(x.view(x.size(0), -1))

class FuseBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(FuseBlock, self).__init__()

        self.fuse0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.fuse00 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.fuse000 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1, padding=3),
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
        x = x1 + x2 + x3
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

# 1000
class arch1000_0(nn.Module):

    def __init__(self):
        super(arch1000_0, self).__init__()

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
        
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse111 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )

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
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)

        return self.head(x.view(x.size(0), -1))

class arch3000_2(nn.Module):

    def __init__(self):
        super(arch3000_2, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1),
        )

        self.proj_fuse_high = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_fuse_low = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
        )

        self.proj_fuse = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
        )
        
        # self.fuse1 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        # self.fuse11 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        # self.fuse111 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        # )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse222 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse333 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse44 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse444 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse55 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse555 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse66 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse666 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.fuse7 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse77 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.fuse777 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        # self.fuse8 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(1),
        #     nn.PReLU(),
        # )
        # self.fuse88 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(1),
        #     nn.PReLU(),
        # )
        # self.fuse888 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(1),
        #     nn.PReLU(),
        # )

        self.head = nn.Sequential(
            nn.Linear(2209, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.PReLU(),
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
        # x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)
        x = self.fuse7(x) + self.fuse77(x) + self.fuse777(x)
        # x = self.fuse8(x) + self.fuse88(x) + self.fuse888(x)

        return self.head(x.view(x.size(0), -1))