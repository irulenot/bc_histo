import torch.nn.functional as F
import torch.nn as nn

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

        self.dropout = nn.Dropout(p=0.0)

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
        self.conv1 = nn.Conv3d(3, 8, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(8, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv3d(8, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(16, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv3d(16, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(32, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv01 = nn.Conv3d(4, 8, kernel_size=1)
        self.bn01 = nn.BatchNorm3d(8, affine=True)
        self.prelu01 = nn.PReLU()

        self.conv02 = nn.Conv3d(8, 16, kernel_size=1)
        self.bn02 = nn.BatchNorm3d(16, affine=True)
        self.prelu02 = nn.PReLU()

        self.conv03 = nn.Conv3d(16, 32, kernel_size=1)
        self.bn03 = nn.BatchNorm3d(32, affine=True)
        self.prelu03 = nn.PReLU()

        self.conv4 = nn.Conv3d(32, 16, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm3d(16, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv3d(16, 8, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm3d(8, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv3d(8, 4, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(4, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv07 = nn.Conv3d(4, 1, kernel_size=2, padding=1, stride=1)
        self.bn07 = nn.BatchNorm3d(1, affine=True)
        self.prelu07 = nn.PReLU()

        self.linear7 = nn.Linear(5 * 64 * 64, 1 * 64 * 64)
        self.bn7 = nn.LayerNorm(64 * 64, elementwise_affine=True)
        self.prelu7 = nn.PReLU()
        self.linear8 = nn.Linear(1 * 64 * 64, 1 * 32 * 32)
        self.bn8 = nn.LayerNorm(32 * 32, elementwise_affine=True)
        self.prelu8 = nn.PReLU()
        self.linear9 = nn.Linear(1 * 32 * 32, 1 * 16 * 16)
        self.bn9 = nn.LayerNorm(16 * 16, elementwise_affine=True)
        self.prelu9 = nn.PReLU()

        self.linear10 = nn.Linear(1 * 16 * 16, 5)

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        # Flatten the input image
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.prelu01(self.bn01(self.conv01(x.permute(0, 2, 1, 3, 4))))
        x = self.dropout(x)
        x = self.prelu02(self.bn02(self.conv02(x)))
        x = self.dropout(x)
        x = self.prelu03(self.bn03(self.conv03(x)))
        x = self.dropout(x)

        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.prelu07(self.bn07(self.conv07(x)))
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
    
class arch24(nn.Module):
    def __init__(self):
        super(arch24, self).__init__()
        self.conv1 = nn.Conv3d(3, 4, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(4, affine=True)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv3d(4, 8, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(8, affine=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv3d(8, 16, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(16, affine=True)
        self.prelu3 = nn.PReLU()

        self.conv01 = nn.Conv3d(2, 4, kernel_size=1)
        self.bn01 = nn.BatchNorm3d(4, affine=True)
        self.prelu01 = nn.PReLU()

        self.conv02 = nn.Conv3d(4, 8, kernel_size=1)
        self.bn02 = nn.BatchNorm3d(8, affine=True)
        self.prelu02 = nn.PReLU()

        self.conv03 = nn.Conv3d(8, 16, kernel_size=1)
        self.bn03 = nn.BatchNorm3d(16, affine=True)
        self.prelu03 = nn.PReLU()

        self.conv4 = nn.Conv3d(16, 8, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm3d(8, affine=True)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv3d(8, 4, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm3d(4, affine=True)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv3d(4, 2, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(2, affine=True)
        self.prelu6 = nn.PReLU()

        self.conv07 = nn.Conv3d(2, 1, kernel_size=2, padding=1, stride=1)
        self.bn07 = nn.BatchNorm3d(1, affine=True)
        self.prelu07 = nn.PReLU()

        self.linear7 = nn.Linear(3 * 126 * 126, 1 * 64 * 64)
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

        x = self.prelu01(self.bn01(self.conv01(x.permute(0, 2, 1, 3, 4))))
        x = self.dropout(x)
        x = self.prelu02(self.bn02(self.conv02(x)))
        x = self.dropout(x)
        x = self.prelu03(self.bn03(self.conv03(x)))
        x = self.dropout(x)

        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.prelu5(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.prelu6(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = self.prelu07(self.bn07(self.conv07(x)))
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