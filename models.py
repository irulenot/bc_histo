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
    
class arch1000(nn.Module):

    def __init__(self):
        super(arch1000, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
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

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)

class arch1000_cond2(nn.Module):

    def __init__(self):
        super(arch1000_cond2, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)

        x1, x3 = x1 * matrix_high, x3 * matrix_low

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond3(nn.Module):

    def __init__(self):
        super(arch1000_cond3, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high2(cond)
        cond_mul_low = self.lin_mul_low2(cond)
        cond_bias_high = self.lin_bias_high2(cond)
        cond_bias_low = self.lin_bias_low2(cond)
        conv_high = self.proj_cond_high2.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low2.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high2, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low2, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond4(nn.Module):

    def __init__(self):
        super(arch1000_cond4, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high2(cond)
        cond_mul_low = self.lin_mul_low2(cond)
        cond_bias_high = self.lin_bias_high2(cond)
        cond_bias_low = self.lin_bias_low2(cond)
        conv_high = self.proj_cond_high2.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low2.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high2, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low2, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high3(cond)
        cond_mul_low = self.lin_mul_low3(cond)
        cond_bias_high = self.lin_bias_high3(cond)
        cond_bias_low = self.lin_bias_low3(cond)
        conv_high = self.proj_cond_high3.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low3.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high3, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low3, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond5(nn.Module):

    def __init__(self):
        super(arch1000_cond5, self).__init__()

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

        self.proj_cond_high3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.proj_cond_high5 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low5 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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
        x01, x03 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        c1 = self.proj_cond_high3(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low3(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high4(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low4(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)
        x11, x13 = x01 * c1, x03 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        c1 = self.proj_cond_high5(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low5(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high6(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low6(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)
        x21, x23 = x11 * c1, x13 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1, x3 = x01 + x11 + x21, x03 + x13 + x23

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)

class arch1000_cond6(nn.Module):

    def __init__(self):
        super(arch1000_cond6, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

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

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

    def forward(self, x, cond):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 + c1, x3 + c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond7(nn.Module):

    def __init__(self):
        super(arch1000_cond7, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

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

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

    def forward(self, x, cond):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond8(nn.Module):

    def __init__(self):
        super(arch1000_cond8, self).__init__()

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

        self.proj_cond_high3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.proj_cond_high5 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low5 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        x1 = self.norm_high_magnitude(x1)
        x3 = self.norm_low_magnitude(x3)

        c1 = self.proj_cond_high3(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low3(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high4(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low4(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)
        x1, x3 = x1 + c1, x3 + c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x2 = self.norm_high_phase(x2)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond9(nn.Module):

    def __init__(self):
        super(arch1000_cond9, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high2(cond)
        cond_mul_low = self.lin_mul_low2(cond)
        cond_bias_high = self.lin_bias_high2(cond)
        cond_bias_low = self.lin_bias_low2(cond)
        conv_high = self.proj_cond_high2.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low2.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high2, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low2, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high3(cond)
        cond_mul_low = self.lin_mul_low3(cond)
        cond_bias_high = self.lin_bias_high3(cond)
        cond_bias_low = self.lin_bias_low3(cond)
        conv_high = self.proj_cond_high3.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low3.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high3, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low3, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        x1 = self.norm_high_magnitude(x1)
        x3 = self.norm_low_magnitude(x3)
        x2 = self.norm_high_phase(x2)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond10(nn.Module):

    def __init__(self):
        super(arch1000_cond10, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)
        x01, x03 = x1 * matrix_high, x3 * matrix_low

        cond_mul_high = self.lin_mul_high2(cond)
        cond_mul_low = self.lin_mul_low2(cond)
        cond_bias_high = self.lin_bias_high2(cond)
        cond_bias_low = self.lin_bias_low2(cond)
        conv_high = self.proj_cond_high2.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low2.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high2, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low2, conv_low, bias=None, stride=1)
        x11, x13 = x01 * matrix_high, x03 * matrix_low

        cond_mul_high = self.lin_mul_high3(cond)
        cond_mul_low = self.lin_mul_low3(cond)
        cond_bias_high = self.lin_bias_high3(cond)
        cond_bias_low = self.lin_bias_low3(cond)
        conv_high = self.proj_cond_high3.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low3.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high3, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low3, conv_low, bias=None, stride=1)
        x21, x23 = x11 * matrix_high, x13 * matrix_low

        x1, x3 = x01 + x11 + x21, x03 + x13 + x23

        x1 = self.norm_high_magnitude(x1)
        x3 = self.norm_low_magnitude(x3)
        x2 = self.norm_high_phase(x2)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond11(nn.Module):

    def __init__(self):
        super(arch1000_cond11, self).__init__()

        # Issues: bigger FFT, scaling up layers
        # Inspect normalization (values, normalize x values, histogram, compare different sizes, compare ranges, with trained model)
        # Just do a single or two 2d

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        shape = (1, 3, 1000, 1000)
        tensor = torch.randn(shape)
        self.param_high1 = nn.Parameter(tensor)
        self.param_low1 = nn.Parameter(tensor)
        self.param_high2 = nn.Parameter(tensor)
        self.param_low2 = nn.Parameter(tensor)
        self.param_high3 = nn.Parameter(tensor)
        self.param_low3 = nn.Parameter(tensor)

        self.lin_mul_high1 = nn.Linear(2, 1)
        self.lin_mul_low1 = nn.Linear(2, 1)
        self.lin_mul_high2 = nn.Linear(2, 1)
        self.lin_mul_low2 = nn.Linear(2, 1)
        self.lin_mul_high3 = nn.Linear(2, 1)
        self.lin_mul_low3 = nn.Linear(2, 1)

        self.lin_bias_high1 = nn.Linear(2, 1)
        self.lin_bias_low1 = nn.Linear(2, 1)
        self.lin_bias_high2 = nn.Linear(2, 1)
        self.lin_bias_low2 = nn.Linear(2, 1)
        self.lin_bias_high3 = nn.Linear(2, 1)
        self.lin_bias_low3 = nn.Linear(2, 1)

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        self.proj_cond_high1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_high3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.proj_cond_low3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)        

        # BatchNorm? (Instance norm, group norm) # LayerNorm? (what about norm that goes across channels rather than height, (reshape?))
        # Dropout after each block
        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

        cond = cond.reshape(2)
        cond_mul_high = self.lin_mul_high1(cond)
        cond_mul_low = self.lin_mul_low1(cond)
        cond_bias_high = self.lin_bias_high1(cond)
        cond_bias_low = self.lin_bias_low1(cond)
        conv_high = self.proj_cond_high1.weight * cond_mul_high + cond_bias_high
        conv_low = self.proj_cond_low1.weight * cond_mul_low + cond_bias_low
        matrix_high = nn.functional.conv2d(self.param_high1, conv_high, bias=None, stride=1)
        matrix_low = nn.functional.conv2d(self.param_low1, conv_low, bias=None, stride=1)
        x1, x3 = x1 * matrix_high, x3 * matrix_low

        x1 = self.norm_high_magnitude(x1)
        x3 = self.norm_low_magnitude(x3)
        x2 = self.norm_high_phase(x2)
        x4 = self.norm_low_phase(x4)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x2 = self.conv_channels_high_phase(x2).squeeze()  # (3x1000x1000) -> (1x1000x1000)
        x3 = self.conv_channels_low_magnitude(x3).squeeze() # (3x1000x1000) -> (1x1000x1000)
        x4 = self.conv_channels_low_phase(x4).squeeze() # (3x1000x1000) -> (1x1000x1000)

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0) # (4x1000x1000)
        x = self.conv_freq(x).view(x.size(0), -1) # (4x1000x1000) -> # (1, 1, 32, 32)
        return self.head(x)
    
class arch1000_cond12(nn.Module):

    def __init__(self):
        super(arch1000_cond12, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

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

        self.proj_cond_high3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

    def forward(self, x, cond):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        c1 = self.proj_cond_high3(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low3(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high4(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low4(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 + c1, x3 + c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond13(nn.Module):

    def __init__(self):
        super(arch1000_cond13, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        self.norm_high_magnitude2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.norm_low_magnitude2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

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

        self.proj_cond_high3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

    def forward(self, x, cond):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        x1 = self.norm_high_magnitude2(x1)
        x3 = self.norm_low_magnitude2(x3)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond14(nn.Module):

    def __init__(self):
        super(arch1000_cond14, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

        self.norm_high_magnitude2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )

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

        self.proj_cond_high3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_high4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )
        self.proj_cond_low4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1000, kernel_size=3, padding=1),
        )

        self.conv_channels_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )               
        self.conv_channels_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=1, stride=1),
            nn.BatchNorm2d(36),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )          

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
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

    def forward(self, x, cond):
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.norm_high_magnitude(x1)
        x2 = self.norm_high_phase(x2)
        x3 = self.norm_low_magnitude(x3)
        x4 = self.norm_low_phase(x4)

        c1 = self.proj_cond_high3(cond.reshape(1, 2, 1, 1)) # (1, 2, 1, 1) -> (1, 1000, 1, 1)
        c2 = self.proj_cond_low3(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high4(c1.permute(0, 2, 1, 3)).squeeze(3) # (1, 1000, 1, 1) -> (1, 1, 1000, 1)
        c2 = self.proj_cond_low4(c2.permute(0, 2, 1, 3)).squeeze(3) # (1000) -> (1000,1000)

        x1, x3 = x1 * c1, x3 * c2 # (1000,1000) * (1000,1000) # Scaling Magnitude

        x1 = self.norm_high_magnitude2(x1)
        x3 = self.norm_low_magnitude2(x3)

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)