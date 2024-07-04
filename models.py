import torch.nn.functional as F
import torch.nn as nn
import torch

class arch2(nn.Module):
    def __init__(self):
        super(arch2, self).__init__()
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

class arch22(nn.Module):
    def __init__(self):
        super(arch22, self).__init__()
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
    
class arch222(nn.Module):
    def __init__(self):
        super(arch222, self).__init__()
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

        self.conv14 = nn.Conv2d(81, 27, kernel_size=1)
        self.bn14 = nn.BatchNorm2d(27, affine=True)
        self.prelu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(27, 9, kernel_size=1)
        self.bn15 = nn.BatchNorm2d(9, affine=True)
        self.prelu15 = nn.PReLU()
        self.conv16 = nn.Conv2d(9, 3, kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Flatten the input image
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x0 = x.clone()

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

        x1 = self.prelu14(self.bn14(self.conv14(x0)))
        x1 = self.dropout(x1)
        x1 = self.prelu15(self.bn15(self.conv15(x1)))
        x1 = self.dropout(x1)
        x1 = self.conv16(x1)

        return x, x1

class arch223(nn.Module):
    def __init__(self):
        super(arch223, self).__init__()
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

class arch23(nn.Module):
    def __init__(self):
        super(arch23, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(32, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv3d(5, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(32, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv04 = nn.Conv3d(32, 64, kernel_size=1)
        self.bn04 = nn.BatchNorm3d(64, affine=True)
        self.prelu04 = nn.PReLU()

        self.conv5 = nn.Conv3d(64, 32, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm3d(32, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv3d(32, 16, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(16, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv3d(16, 8, kernel_size=3, padding=1, stride=2)
        self.bn7 = nn.BatchNorm3d(8, affine=True)
        self.prelu7 = nn.PReLU()

        self.conv8 = nn.Conv3d(8, 4, kernel_size=3, padding=1, stride=2)
        self.bn8 = nn.BatchNorm3d(4, affine=True)
        self.prelu8 = nn.PReLU()

        self.conv9 = nn.Conv3d(4, 2, kernel_size=3, padding=1, stride=2)
        self.bn9 = nn.BatchNorm3d(2, affine=True)
        self.prelu9 = nn.PReLU()

        self.conv10 = nn.Conv3d(2, 1, kernel_size=3, padding=1, stride=2)
        self.bn10 = nn.BatchNorm3d(1, affine=True)
        self.prelu10 = nn.PReLU()

        self.linear10 = nn.Linear(1 * 8 * 8, 5)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.prelu2(self.bn2(self.conv2(x.permute(0, 2, 1, 3, 4))))
        x = self.dropout(x)

        x = self.prelu04(self.bn04(self.conv04(x.permute(0, 2, 1, 3, 4))))
        x = self.dropout(x)
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.prelu7(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = self.prelu8(self.bn8(self.conv8(x)))
        x = self.dropout(x)
        x = self.prelu9(self.bn9(self.conv9(x)))
        x = self.dropout(x)
        x = self.prelu10(self.bn10(self.conv10(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.linear10(x)
        return x

class arch24(nn.Module):
    def __init__(self):
        super(arch24, self).__init__()
        self.conv1 = nn.Conv2d(6, 12, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(12, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(12, 24, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(24, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(24, 48, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(48, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(48, 96, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(96, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv04 = nn.Conv2d(96, 48, kernel_size=3, padding=1, stride=2)
        self.bn04 = nn.BatchNorm2d(48, affine=True)
        self.prelu04 = nn.PReLU()

        self.conv5 = nn.Conv2d(48, 24, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(24, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(24, 12, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(12, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(12, 6, kernel_size=3, padding=1, stride=2)
        self.bn7 = nn.BatchNorm2d(6, affine=True)
        self.prelu7 = nn.PReLU()

        self.conv8 = nn.Conv2d(6, 1, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(1, affine=True)
        self.prelu8 = nn.PReLU()

        self.linear17 = nn.Linear(1 * 63 * 63, 1 * 32 * 32)
        self.bn17 = nn.LayerNorm(32 * 32, elementwise_affine=True)
        self.prelu17 = nn.PReLU()
        self.linear18 = nn.Linear(1 * 32 * 32, 1 * 16 * 16)
        self.bn18 = nn.LayerNorm(16 * 16, elementwise_affine=True)
        self.prelu18 = nn.PReLU()
        self.linear19 = nn.Linear(1 * 16 * 16, 1 * 8 * 8)
        self.bn19 = nn.LayerNorm(8 * 8, elementwise_affine=True)
        self.prelu19 = nn.PReLU()

        self.linear10 = nn.Linear(1 * 8 * 8, 5)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Flatten the input image
        x = x.reshape(x.size(0), -1, 1000, 1000)

        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = self.prelu04(self.bn04(self.conv04(x)))
        x = self.dropout(x)
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.prelu7(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = self.prelu8(self.bn8(self.conv8(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.prelu17(self.bn17(self.linear17(x)))
        x = self.dropout(x)
        x = self.prelu18(self.bn18(self.linear18(x)))
        x = self.dropout(x)
        x = self.prelu19(self.bn19(self.linear19(x)))
        x = self.dropout(x)
        
        x = self.linear10(x)
        return x

class arch242(nn.Module):
    def __init__(self):
        super(arch242, self).__init__()
        self.conv1 = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(4, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(8, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(16, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(32, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(64, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(128, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv3d(128, 128, kernel_size=3)
        self.bn7 = nn.BatchNorm3d(128, affine=True)
        self.prelu7 = nn.PReLU()

        self.conv21 = nn.Conv2d(128, 64, kernel_size=3, stride=2)
        self.bn21 = nn.BatchNorm2d(64, affine=True)
        self.prelu21 = nn.PReLU()

        self.conv22 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.bn22 = nn.BatchNorm2d(32, affine=True)
        self.prelu22 = nn.PReLU()

        self.conv23 = nn.Conv2d(32, 16, kernel_size=3, stride=2)
        self.bn23 = nn.BatchNorm2d(16, affine=True)
        self.prelu23 = nn.PReLU()

        self.conv24 = nn.Conv2d(16, 8, kernel_size=3, stride=2)
        self.bn24 = nn.BatchNorm2d(8, affine=True)
        self.prelu24 = nn.PReLU()

        self.conv25 = nn.Conv2d(8, 4, kernel_size=3, stride=2)
        self.bn25 = nn.BatchNorm2d(4, affine=True)
        self.prelu25 = nn.PReLU()

        self.conv26 = nn.Conv2d(4, 2, kernel_size=3, stride=2)
        self.bn26 = nn.BatchNorm2d(2, affine=True)
        self.prelu26 = nn.PReLU()

        self.conv27 = nn.Conv2d(2, 1, kernel_size=3, stride=2)
        self.bn27 = nn.BatchNorm2d(1, affine=True)
        self.prelu27 = nn.PReLU()

        self.linear33 = nn.Linear(36, 5)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
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
        x = self.prelu7(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = x.squeeze(2)

        x = self.prelu21(self.bn21(self.conv21(x)))
        x = self.dropout(x)
        x = self.prelu22(self.bn22(self.conv22(x)))
        x = self.dropout(x)
        x = self.prelu23(self.bn23(self.conv23(x)))
        x = self.dropout(x)
        x = self.prelu24(self.bn24(self.conv24(x)))
        x = self.dropout(x)
        x = self.prelu25(self.bn25(self.conv25(x)))
        x = self.dropout(x)
        x = self.prelu26(self.bn26(self.conv26(x)))
        x = self.dropout(x)
        x = self.prelu27(self.bn27(self.conv27(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return self.linear33(x)