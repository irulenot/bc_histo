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
    
class arch2000(nn.Module):

    def __init__(self):
        super(arch2000, self).__init__()

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
    
class arch1000_norm(nn.Module):

    def __init__(self):
        super(arch1000_norm, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=27, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=27, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=27, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=27, out_channels=9, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1),
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
    
class arch1000_phase(nn.Module):

    def __init__(self):
        super(arch1000_phase, self).__init__()

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
        )
        self.norm_low_phase = nn.Sequential(
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

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
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

        x1 = x1 + x2
        x3 = x3 + x4

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()

        x = torch.stack([x1, x3]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_gray(nn.Module):

    def __init__(self):
        super(arch1000_gray, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
        )
        self.norm_high_phase = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
        )
        self.norm_low_phase = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
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
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
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
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1),
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
        x1 = torch.abs(x[0]).unsqueeze(0)
        x2 = torch.angle(x[0]).unsqueeze(0)
        x3 = torch.abs(x[1]).unsqueeze(0)
        x4 = torch.angle(x[1]).unsqueeze(0)

        x1 = self.norm_high_magnitude(x1).squeeze()
        x2 = self.norm_high_phase(x2).squeeze()
        x3 = self.norm_low_magnitude(x3).squeeze()
        x4 = self.norm_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_mul(nn.Module):

    def __init__(self):
        super(arch1000_mul, self).__init__()

        self.matrix_high_magnitude1 = nn.Parameter(torch.randn(1000, 1000))
        self.matrix_low_magnitude1 = nn.Parameter(torch.randn(1000, 1000))
        self.matrix_high_magnitude2 = nn.Parameter(torch.randn(1000, 1000))
        self.matrix_low_magnitude2 = nn.Parameter(torch.randn(1000, 1000))
        self.matrix_high_magnitude3 = nn.Parameter(torch.randn(1000, 1000))
        self.matrix_low_magnitude3 = nn.Parameter(torch.randn(1000, 1000))

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

        x1 = x1 * self.matrix_high_magnitude1
        x3 = x3 * self.matrix_low_magnitude1
        x1 = x1 * self.matrix_high_magnitude2
        x3 = x3 * self.matrix_low_magnitude2
        x1 = x1 * self.matrix_high_magnitude3
        x3 = x3 * self.matrix_low_magnitude3

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class ElementWiseMultiplier(nn.Module):
    def __init__(self, input_dim):
        super(ElementWiseMultiplier, self).__init__()
        # Initialize a learnable parameter of shape [input_dim]
        self.learnable_parameter = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # Perform element-wise multiplication
        return x * self.learnable_parameter

class arch1000_mul2(nn.Module):

    def __init__(self):
        super(arch1000_mul2, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
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
    
class arch1000_mul3(nn.Module):

    def __init__(self):
        super(arch1000_mul3, self).__init__()

        self.norm_high_magnitude = nn.Sequential(
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
        )
        self.norm_low_magnitude = nn.Sequential(
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1),
            ElementWiseMultiplier(1000),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
        )
        self.norm_phase = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )

        self.combine_high = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=1),
        )
        self.combine_low = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=1),
        )

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
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

        x1 = self.norm_high_magnitude(x1).squeeze()
        x3 = self.norm_low_magnitude(x3).squeeze()
        x2, x4 = self.norm_phase(x2).squeeze(), self.norm_phase(x4).squeeze()

        x1 = torch.stack([x1, x2]).unsqueeze(0)
        x3 = torch.stack([x3, x4]).unsqueeze(0)

        x1 = self.combine_high(x1).squeeze()
        x3 = self.combine_low(x3).squeeze()

        x = torch.stack([x1, x3]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_comb(nn.Module):

    def __init__(self):
        super(arch1000_comb, self).__init__()

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

        self.conv_channels_low = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=9, kernel_size=1, stride=1),
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
        self.conv_channels_high = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=9, kernel_size=1, stride=1),
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
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
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

        x1 = torch.stack([x1, x2]).unsqueeze(0)
        x3 = torch.stack([x3, x4]).unsqueeze(0)
        x1 = self.conv_channels_high(x1).squeeze()
        x3 = self.conv_channels_low(x3).squeeze()

        x = torch.stack([x1, x3]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond(nn.Module):

    def __init__(self):
        super(arch1000_cond, self).__init__()

        self.proj_cond_high = nn.Sequential(
            nn.Linear(2, 1),
        )
        self.proj_cond_low = nn.Sequential(
            nn.Linear(2, 1),
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

        c1 = self.proj_cond_high(cond)
        c2 = self.proj_cond_low(cond)

        x1, x3 = x1 * c1, x3 * c2

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
    
class arch1000_cond2(nn.Module):

    def __init__(self):
        super(arch1000_cond2, self).__init__()

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

        c1 = self.proj_cond_high1(cond.reshape(1, 2, 1, 1))
        c2 = self.proj_cond_low1(cond.reshape(1, 2, 1, 1))
        c1 = self.proj_cond_high2(c1.permute(0, 2, 1, 3)).squeeze(3)
        c2 = self.proj_cond_low2(c2.permute(0, 2, 1, 3)).squeeze(3)

        x1, x3 = x1 * c1, x3 * c2

        x1 = self.conv_channels_high_magnitude(x1).squeeze()
        x2 = self.conv_channels_high_phase(x2).squeeze()
        x3 = self.conv_channels_low_magnitude(x3).squeeze()
        x4 = self.conv_channels_low_phase(x4).squeeze()

        x = torch.stack([x1, x2, x3, x4]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)
    
class arch1000_cond3(nn.Module):

    def __init__(self):
        super(arch1000_cond3, self).__init__()

        ones_tensor = torch.ones((1000,1000), dtype=torch.float)
        self.matrix_high_magnitude = nn.Parameter(ones_tensor)
        self.matrix_low_magnitude = nn.Parameter(ones_tensor)

        self.proj_cond_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.proj_cond_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

        self.conv_channels_high = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1, stride=1),
            nn.BatchNorm2d(12),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1),
            nn.BatchNorm2d(12),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.conv_channels_low = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1, stride=1),
            nn.BatchNorm2d(12),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1),
            nn.BatchNorm2d(12),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(0.1),
        )                  

        self.conv_freq = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
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

        c = cond[0] * cond[1]
        matrix_high_magnitude = self.matrix_high_magnitude * c
        matrix_low_magnitude = self.matrix_low_magnitude * c
        c1 = self.proj_cond_high(matrix_high_magnitude.unsqueeze(0).unsqueeze(0))
        c2 = self.proj_cond_low(matrix_low_magnitude.unsqueeze(0).unsqueeze(0))

        x1, x3 = x1 * c1, x3 * c2

        x1 = torch.cat([x1, x2], dim=1)
        x3 = torch.cat([x3, x4], dim=1)

        x1 = self.conv_channels_high(x1).squeeze()
        x3 = self.conv_channels_low(x3).squeeze()

        x = torch.stack([x1, x3]).unsqueeze(0)
        x = self.conv_freq(x).view(x.size(0), -1)
        return self.head(x)