import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import math

class arch0(nn.Module):
    def __init__(self):
        super(arch0, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(9, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(9, 27, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(27, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(27, 81, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(81, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(81, 27, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(27, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(27, 9, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(9, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(9, 3, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(3, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv0 = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1)
        self.bn0 = nn.BatchNorm2d(1, affine=True)
        self.prelu0 = nn.PReLU()

        self.linear7 = nn.Linear(1 * 128 * 128, 1 * 64 * 64)
        self.bn7 = nn.LayerNorm(64 * 64, elementwise_affine=True)
        self.prelu7 = nn.PReLU()
        self.linear8 = nn.Linear(1 * 64 * 64, 1 * 32 * 32)
        self.bn8 = nn.LayerNorm(32 * 32, elementwise_affine=True)
        self.prelu8 = nn.PReLU()
        self.linear9 = nn.Linear(1 * 32 * 32, 1 * 16 * 16)
        self.bn9 = nn.LayerNorm(16 * 16, elementwise_affine=True)
        self.prelu9 = nn.PReLU()

        self.linear10 = nn.Linear(1 * 16 * 16, 5)

    def forward(self, x):
        # Flatten the input image
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.prelu0(self.bn0(self.conv0(x)))

        x = x.view(x.size(0), -1)
        x = self.prelu7(self.bn7(self.linear7(x)))
        x = self.prelu8(self.bn8(self.linear8(x)))
        x = self.prelu9(self.bn9(self.linear9(x)))
        
        x = self.linear10(x)
        return x

class arch00(nn.Module):
    def __init__(self):
        super(arch00, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(9, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(9, 27, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(27, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(27, 81, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(81, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(81, 27, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(27, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(27, 9, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(9, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(9, 3, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(3, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv0 = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1)
        self.bn0 = nn.BatchNorm2d(1, affine=True)
        self.prelu0 = nn.PReLU()

        self.linear7 = nn.Linear(1 * 125 * 125, 1 * 64 * 64)
        self.bn7 = nn.LayerNorm(64 * 64, elementwise_affine=True)
        self.prelu7 = nn.PReLU()
        self.linear8 = nn.Linear(1 * 64 * 64, 1 * 32 * 32)
        self.bn8 = nn.LayerNorm(32 * 32, elementwise_affine=True)
        self.prelu8 = nn.PReLU()
        self.linear9 = nn.Linear(1 * 32 * 32, 1 * 16 * 16)
        self.bn9 = nn.LayerNorm(16 * 16, elementwise_affine=True)
        self.prelu9 = nn.PReLU()

        self.linear10 = nn.Linear(1 * 16 * 16, 5)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Flatten the input image
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.prelu0(self.bn0(self.conv0(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.prelu7(self.bn7(self.linear7(x)))
        x = self.dropout(x)
        x = self.prelu8(self.bn8(self.linear8(x)))
        x = self.dropout(x)
        x = self.prelu9(self.bn9(self.linear9(x)))
        x = self.dropout(x)
        
        x = self.linear10(x)
        return x

#####################################

class ProjBlock(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0):
        super(ProjBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout(x)
        return x

class arch1000(nn.Module):

    def __init__(self):
        super(arch1000, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(), ProjBlock()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(), ProjBlock()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock(), ProjBlock()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock(), ProjBlock()

        in_channels, out_channels = 12, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))
    
class arch1000_cond(nn.Module):

    def __init__(self):
        super(arch1000_cond, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d
        self.proj_cond_high1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        in_channels, out_channels = 3, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.PReLU(),
            nn.Linear(32, 5),
        )

    # Investigate potential ranges depending on resolution (static operation)
    # Before conv_freq combine mag and phase
    # Consistency in resolutions for cond
    def forward(self, x, cond): # cond: (2)
        x1 = torch.abs(x[0]).unsqueeze(0) # (1000,1000)   # high freq Magnitude
        x2 = torch.angle(x[0]).unsqueeze(0) # (1000,1000) # Phase
        x3 = torch.abs(x[1]).unsqueeze(0) # (1000,1000)  # low freq Magnitude
        x4 = torch.angle(x[1]).unsqueeze(0) # (1000,1000)  # Phase

        # (2) = (50000, 10000) = (10000, 50000)
        # (2) -> (1000,1000)
        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class ProjBlock2(nn.Module):
    def __init__(self, in_channels=24, out_channels=24, kernel_size=1, stride=1, padding=0):
        super(ProjBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.dropout(x)
        return x

class arch1000_1(nn.Module):

    def __init__(self):
        super(arch1000_1, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock2(), ProjBlock2()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock2(), ProjBlock2()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock2(), ProjBlock2()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock2(), ProjBlock2()

        in_channels, out_channels = 24, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_2(nn.Module):

    def __init__(self):
        super(arch1000_2, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock2(), ProjBlock2()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock2(), ProjBlock2()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock2(), ProjBlock2()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock2(), ProjBlock2()
        self.proj_mag_high4, self.proj_mag_low4 = ProjBlock2(), ProjBlock2()

        in_channels, out_channels = 24, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion3 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high4(x1)
        x02 = self.proj_mag_low4(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0
        x0 = self.fusion3(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_3(nn.Module):

    def __init__(self):
        super(arch1000_3, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=18, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=18, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=18, out_channels=24, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=18, out_channels=24, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock2(), ProjBlock2()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock2(), ProjBlock2()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock2(), ProjBlock2()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock2(), ProjBlock2()
        self.proj_mag_high4, self.proj_mag_low4 = ProjBlock2(), ProjBlock2()

        in_channels, out_channels = 24, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock2(in_channels=in_channels, out_channels=out_channels), ProjBlock2(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion3 = ProjBlock2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high4(x1)
        x02 = self.proj_mag_low4(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0
        x0 = self.fusion3(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_4(nn.Module):

    def __init__(self):
        super(arch1000_4, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(), ProjBlock()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(), ProjBlock()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock(), ProjBlock()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock(), ProjBlock()
        self.proj_mag_high4, self.proj_mag_low4 = ProjBlock(), ProjBlock()

        in_channels, out_channels = 12, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion3 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high4(x1)
        x02 = self.proj_mag_low4(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0
        x0 = self.fusion3(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_5(nn.Module):

    def __init__(self):
        super(arch1000_5, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(), ProjBlock()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(), ProjBlock()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock(), ProjBlock()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock(), ProjBlock()
        self.proj_mag_high4, self.proj_mag_low4 = ProjBlock(), ProjBlock()
        self.proj_mag_high5, self.proj_mag_low5 = ProjBlock(), ProjBlock()

        in_channels, out_channels = 12, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion3 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion4 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high4(x1)
        x02 = self.proj_mag_low4(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high5(x1)
        x02 = self.proj_mag_low5(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0
        x0 = self.fusion3(x)
        x = x + x0
        x0 = self.fusion4(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class arch2000(nn.Module):

    def __init__(self):
        super(arch2000, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
        )

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(), ProjBlock()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(), ProjBlock()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock(), ProjBlock()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock(), ProjBlock()
        self.proj_mag_high4, self.proj_mag_low4 = ProjBlock(), ProjBlock()
        self.proj_mag_high5, self.proj_mag_low5 = ProjBlock(), ProjBlock()

        in_channels, out_channels = 12, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion3 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion4 = ProjBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 32),
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

        x01 = self.proj_mag_high1(x1)
        x02 = self.proj_mag_low1(x2)
        x03 = self.proj_phase_high1(x3)
        x04 = self.proj_phase_low1(x4)
        x1, x2 = x1 + x01, x2 + x02
        x3, x4 = x3 + x03, x4 + x04

        x01 = self.proj_mag_high2(x1)
        x02 = self.proj_mag_low2(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high3(x1)
        x02 = self.proj_mag_low3(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high4(x1)
        x02 = self.proj_mag_low4(x2)
        x1, x2 = x1 + x01, x2 + x02
        x01 = self.proj_mag_high5(x1)
        x02 = self.proj_mag_low5(x2)
        x1, x2 = x1 + x01, x2 + x02

        x1, x2 = self.gray_mag_high(x1), self.gray_mag_low(x2)
        x3, x4 = self.gray_phase_high(x3), self.gray_phase_low(x4)

        x = x1 + x2 + x3 + x4
        x0 = self.fusion(x)
        x = x + x0
        x0 = self.fusion1(x)
        x = x + x0
        x0 = self.fusion2(x)
        x = x + x0
        x0 = self.fusion3(x)
        x = x + x0
        x0 = self.fusion4(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))