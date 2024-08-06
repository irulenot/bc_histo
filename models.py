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

class arch1000_1(nn.Module):

    def __init__(self):
        super(arch1000_1, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )

        in_channels, out_channels = 64, 64
        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels = 64, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion1 = ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion2 = ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion3 = ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion4 = ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion5 = ProjBlock(in_channels=in_channels, out_channels=out_channels)

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
        x0 = self.fusion3(x)
        x = x + x0
        x0 = self.fusion4(x)
        x = x + x0
        x0 = self.fusion5(x)
        x = x + x0

        x = self.down(x)
        return self.head(x.view(x.size(0), -1))

class ProjBlock2(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0):
        super(ProjBlock2, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn1(x0)
        x0 = self.prelu1(x0)
        x = x + x0
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.prelu1(x0)
        x = x + x0
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout(x)
        return x

class arch1000_2(nn.Module):

    def __init__(self):
        super(arch1000_2, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),
        )

        in_channels, out_channels = 64, 64
        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock(in_channels=in_channels, out_channels=out_channels), ProjBlock(in_channels=in_channels, out_channels=out_channels)

        self.fusion = ProjBlock2(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fusion1 = ProjBlock2(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fusion2 = ProjBlock2(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.fusion3 = ProjBlock2(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.fusion4 = ProjBlock2(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1)

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

        x1 = self.proj_mag_high1(x1)
        x2 = self.proj_mag_low1(x2)
        x3 = self.proj_phase_high1(x3)
        x4 = self.proj_phase_low1(x4)

        x = x1 + x2 + x3 + x4
        x = self.fusion(x)
        x = self.fusion1(x)
        x = self.fusion2(x)
        x = self.fusion3(x)
        x = self.fusion4(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_3(nn.Module):

    def __init__(self):
        super(arch1000_3, self).__init__()

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
            nn.Linear(32, 6),
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

class ProjBlock3(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0):
        super(ProjBlock3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        d_k = math.sqrt(x.size(2))
        Q, K, V = self.conv1(x), self.conv2(x), self.conv3(x)
        scores = torch.matmul(Q, K.transpose(2, 3))  # (batch_size, seq_len, seq_len)
        scaled_scores = scores / (d_k ** 0.5)  # Scaling step
        attention_weights = F.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        x = torch.matmul(attention_weights, V)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
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

        self.proj_mag_high1, self.proj_mag_low1 = ProjBlock3(), ProjBlock3()
        self.proj_phase_high1, self.proj_phase_low1 = ProjBlock3(), ProjBlock3()
        
        self.proj_mag_high2, self.proj_mag_low2 = ProjBlock3(), ProjBlock3()
        self.proj_mag_high3, self.proj_mag_low3 = ProjBlock3(), ProjBlock3()

        in_channels, out_channels = 12, 1
        self.gray_mag_high, self.gray_mag_low = ProjBlock3(in_channels=in_channels, out_channels=out_channels), ProjBlock3(in_channels=in_channels, out_channels=out_channels)
        self.gray_phase_high, self.gray_phase_low = ProjBlock3(in_channels=in_channels, out_channels=out_channels), ProjBlock3(in_channels=in_channels, out_channels=out_channels)

        in_channels, out_channels, kernel_size, padding = 1, 1, 3, 1
        self.fusion = ProjBlock3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion1 = ProjBlock3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.fusion2 = ProjBlock3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

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

class arch1000_6(nn.Module):

    def __init__(self):
        super(arch1000_6, self).__init__()

        self.conv1 = ProjBlock(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = ProjBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ProjBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = ProjBlock(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv5 = ProjBlock(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
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

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_7(nn.Module):

    def __init__(self):
        super(arch1000_7, self).__init__()

        self.conv1 = ProjBlock(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = ProjBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv21 = ProjBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ProjBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv31 = ProjBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = ProjBlock(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv41 = ProjBlock(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv5 = ProjBlock(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv51 = ProjBlock(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
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

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv1(x)
        x0 = self.conv2(x)
        x = self.conv21(x) + x0
        x0 = self.conv3(x)
        x = self.conv31(x) + x0
        x0 = self.conv4(x)
        x = self.conv41(x) + x0
        x0 = self.conv5(x)
        x = self.conv51(x) + x0
        return self.head(x.view(x.size(0), -1))

class arch1000_8(nn.Module):

    def __init__(self):
        super(arch1000_8, self).__init__()

        self.conv1 = ProjBlock3(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = ProjBlock3(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ProjBlock3(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = ProjBlock3(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv5 = ProjBlock3(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv6 = ProjBlock3(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv7 = ProjBlock3(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
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

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return self.head(x.view(x.size(0), -1))

class ProjBlock4(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0, num_heads=2):
        super(ProjBlock4, self).__init__()
        self.num_heads = num_heads
        self.d_k = out_channels // num_heads  # Dimension of each head

        # Ensure out_channels is divisible by num_heads
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        # Define the convolutional layers for Q, K, V
        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_k = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_v = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        # Define the output linear layer
        self.fc_out = nn.Linear(out_channels, out_channels)

        self.dropout = nn.Dropout(0.2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Compute Q, K, V
        Q = self.conv_q(x).view(batch_size, self.num_heads, self.d_k, H*W).transpose(2, 3)  # (batch_size, num_heads, d_k, seq_len)
        K = self.conv_k(x).view(batch_size, self.num_heads, self.d_k, H*W)  # (batch_size, num_heads, d_k, seq_len)
        V = self.conv_v(x).view(batch_size, self.num_heads, self.d_k, H*W)  # (batch_size, num_heads, d_k, seq_len)

        # Compute scores
        scores = torch.matmul(Q, K.transpose(2, 3))  # (batch_size, num_heads, seq_len, seq_len)
        scaled_scores = scores / math.sqrt(self.d_k)  # Scaling step
        attention_weights = F.softmax(scaled_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Compute the weighted sum
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, d_k, seq_len)

        # Concatenate heads and pass through linear layer
        attention_output = attention_output.transpose(2, 3).contiguous().view(batch_size, -1, H, W)  # (batch_size, out_channels, H, W)
        x = self.fc_out(attention_output)
        
        # Apply normalization, activation, and dropout
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class arch1000_9(nn.Module):

    def __init__(self):
        super(arch1000_9, self).__init__()

        self.conv1 = ProjBlock4(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = ProjBlock4(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ProjBlock4(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = ProjBlock4(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv5 = ProjBlock4(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv6 = ProjBlock4(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv7 = ProjBlock4(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
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

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return self.head(x.view(x.size(0), -1))