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
    def __init__(self, in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0, linear=False):
        super(ProjBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear = linear

    def forward(self, x):
        x = self.conv1(x)
        if not self.linear:
            x = self.bn1(x)
            x = self.prelu1(x)
        x = self.conv2(x)
        if not self.linear:
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
            nn.BatchNorm2d(1), #1x1x1000x1000
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1), #1x1x500x500
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1), #1x1x250x250
            nn.PReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1), #1x1x128x128
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

    def forward(self, x): # 2x3x1000x1000
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.abs(x[1]).unsqueeze(0)
        x3 = torch.angle(x[0]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0) 

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

        self.proj_fuse_high = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_fuse_low = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1),
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

        x1, x2 = torch.cat([x1, x3], dim=1), torch.cat([x2, x4], dim=1)
        x1, x2 = self.proj_fuse_high(x1), self.proj_fuse_low(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj_fuse(x)
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)

        return self.head(x.view(x.size(0), -1))
    
class arch1000_4(nn.Module):

    def __init__(self):
        super(arch1000_4, self).__init__()

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

class arch1000_5(nn.Module):

    def __init__(self):
        super(arch1000_5, self).__init__()

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
        x1, x2, x3 = self.fuse1(x), self.fuse11(x), self.fuse111(x)
        x1, x2, x3 = self.fuse2(x1), self.fuse22(x2), self.fuse222(x3)
        x1, x2, x3 = self.fuse3(x1), self.fuse33(x2), self.fuse333(x3)
        x1, x2, x3 = self.fuse4(x1), self.fuse44(x2), self.fuse444(x3)
        x1, x2, x3 = self.fuse5(x1), self.fuse55(x2), self.fuse555(x3)
        x1, x2, x3 = self.fuse6(x1), self.fuse66(x2), self.fuse666(x3)
        x = x1 + x2 + x3

        return self.head(x.view(x.size(0), -1))
    
class arch1000_6(nn.Module):

    def __init__(self):
        super(arch1000_6, self).__init__()

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
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        
        self.proj_fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )

        self.head = nn.Sequential(
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

        x1, x2 = x1 + x1.min(), x2 + x2.min()

        x1 = self.proj_mag_high(x1)
        x2 = self.proj_mag_low(x2)
        x3 = self.proj_phase_high(x3)
        x4 = self.proj_phase_low(x4)

        x1, x2 = x1 * x3, x2 * x4
        x1, x2 = self.proj_fuse_high(x1), self.proj_fuse_low(x2)
        x = x1 * x2
        x = self.proj_fuse(x)
        x = self.proj_fuse1(x)

        x = x.view(1, 128, 1024).permute(1, 0, 2)
        x = self.transformer_encoder(x).permute(1, 0, 2).reshape(1, 128, 32, 32)
        drop = nn.Dropout(0.2)
        x = drop(x)
        x = self.down(x)
        x = drop(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_7(nn.Module):

    def __init__(self):
        super(arch1000_7, self).__init__()

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
        x1 = torch.abs(x[:, 0])
        x2 = torch.abs(x[:, 1])
        x3 = torch.angle(x[:, 0])
        x4 = torch.angle(x[:, 1]) # 3x1000x1000

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

class arch2000_8(nn.Module):

    def __init__(self):
        super(arch2000_8, self).__init__()

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
            nn.Linear(2048, 1024),
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
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x)

        return self.head(x.view(x.size(0), -1))

class arch1000_9(nn.Module):

    def __init__(self):
        super(arch1000_9, self).__init__()

        self.proj_mag_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_mag_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
        )
        self.proj_phase_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
        )

        self.proj_high = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        )
        self.proj_low = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
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

        x1, x2 = self.proj_high(x1), self.proj_low(x2)
        x = x1 * x2

        x = self.fusion(x)
        return self.head(x.view(x.size(0), -1))

class arch1000_10(nn.Module):

    def __init__(self):
        super(arch1000_10, self).__init__()

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

        self.proj_fuse_high = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1),
        )
        self.proj_fuse_low = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1),
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

        x1, x2 = torch.cat([x1, x3], dim=1), torch.cat([x2, x4], dim=1)
        x1, x2 = self.proj_fuse_high(x1), self.proj_fuse_low(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj_fuse(x)
        drop = nn.Dropout(0.2)
        x = self.fuse1(x) + self.fuse11(x) + self.fuse111(x) + self.fuse1111(x)
        x = drop(x)
        x = self.fuse2(x) + self.fuse22(x) + self.fuse222(x) + self.fuse2222(x)
        x = drop(x)
        x = self.fuse3(x) + self.fuse33(x) + self.fuse333(x) + self.fuse3333(x)
        x = drop(x)
        x = self.fuse4(x) + self.fuse44(x) + self.fuse444(x) + self.fuse4444(x)
        x = drop(x)
        x = self.fuse5(x) + self.fuse55(x) + self.fuse555(x) + self.fuse5555(x)
        x = drop(x)
        x = self.fuse6(x) + self.fuse66(x) + self.fuse666(x) + self.fuse6666(x)
        x = drop(x)

        return self.head(x.view(x.size(0), -1))

class arch1000_11(nn.Module):

    def __init__(self):
        super(arch1000_11, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1, stride=1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
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
        x1 = torch.abs(x[0])
        x2 = torch.abs(x[1])
        x3 = torch.angle(x[0])
        x4 = torch.angle(x[1])

        x = torch.cat([x1, x2, x3, x4], dim=0).unsqueeze(0)
        x = self.conv(x)
        return self.head(x.view(x.size(0), -1))