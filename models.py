import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2

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
    
class KernelModule(nn.Module):
    def __init__(self, size=2000):
        super(KernelModule, self).__init__()
        self.magnitude_kernel = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel = nn.Parameter(torch.randn(1, size, size))
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.1)
        self.reduction = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, fin=False):
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        magnitude = magnitude * self.magnitude_kernel
        phase = phase * self.phase_kernel

        combined = magnitude * torch.exp(1j * phase)
        combined = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        x = self.norm(combined.unsqueeze(1))
        x = self.activation(x)
        x = self.reduction(x)

        x = self.dropout(x).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x

class arch3(nn.Module):

    def __init__(self, num_kernels=4):
        super(arch3, self).__init__()
        self.num_kernels = num_kernels
        sizes = [2000, 1000, 500, 250]
        kernels = []
        for k in range(num_kernels):
            kernels.append(KernelModule(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(125 * 125, 125)
        self.fc2 = nn.Linear(125, 5)

    def forward(self, x):
        for i, kernel in enumerate(self.kernels):
            if i < self.num_kernels-1:
                x = kernel(x)
            else:
                x = kernel(x, fin=True)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class KernelModule2(nn.Module):
    def __init__(self, size=2000, num_kernels=1):
        super(KernelModule2, self).__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ParameterList()
        self.magnitude_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        for k in range(num_kernels):
            magnitude_kernel = nn.Parameter(torch.randn(1, size, size))
            phase_kernel = nn.Parameter(torch.randn(1, size, size))
            self.kernels.extend([magnitude_kernel, phase_kernel])
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.1)
        self.reduction = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, fin=False):
        magnitude0 = torch.abs(x)
        phase0 = torch.angle(x)

        magnitude = magnitude0.clone()
        phase = phase0.clone()
        for k in range(self.num_kernels):
            magnitude += magnitude0 * self.kernels[2*k]
            phase += phase0 * self.kernels[2*k+1]
        magnitude += self.magnitude_kernel_bias
        phase += self.phase_kernel_bias
        
        combined = magnitude * torch.exp(1j * phase)
        combined = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        x = self.norm(combined.unsqueeze(1))
        x = self.activation(x)
        x = self.reduction(x)

        x = self.dropout(x).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x

class KernelModule3(nn.Module):
    def __init__(self, size=2000, num_kernels=1):
        super(KernelModule3, self).__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ParameterList()
        self.magnitude_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        for k in range(num_kernels):
            magnitude_kernel = nn.Parameter(torch.randn(1, size, size))
            phase_kernel = nn.Parameter(torch.randn(1, size, size))
            self.kernels.extend([magnitude_kernel, phase_kernel])
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.1)
        self.reduction = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, fin=False):
        magnitude0 = torch.abs(x)
        phase0 = torch.angle(x)

        magnitudes = []
        phases = []
        for k in range(self.num_kernels):
            magnitudes.append(magnitude0 * self.kernels[2*k])
            phases.append(phase0 * self.kernels[2*k+1])
        magnitude = torch.sum(torch.stack(magnitudes), dim=1) + self.magnitude_kernel_bias
        phase = torch.sum(torch.stack(phases), dim=1) + self.phase_kernel_bias
        
        combined = magnitude * torch.exp(1j * phase)
        combined = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        x = self.norm(combined.unsqueeze(1))
        x = self.activation(x)
        x = self.reduction(x)

        x = self.dropout(x).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x

class KernelModule0(nn.Module):
    def __init__(self, size=2000):
        super(KernelModule0, self).__init__()
        self.magnitude_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel_bias = nn.Parameter(torch.randn(1, size, size))
        self.magnitude_kernel = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel = nn.Parameter(torch.randn(1, size, size))
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1))
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(0.1)
        self.reduction = nn.Conv2d(1, 1, kernel_size=2, stride=2)

    def forward(self, x, fin=False, reduce=False):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        magnitude = magnitude * self.magnitude_kernel
        phase = phase * self.phase_kernel
        # magnitude = magnitude + self.magnitude_kernel_bias
        # phase = phase + self.phase_kernel_bias
        
        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        # x_shape = x.shape
        # x = x.view(x_shape[0], -1)
        # x = F.normalize(x, p=2, dim=1)
        # x = x * self.scale + self.bias
        # x = x.reshape(x_shape) 
        # x = self.activation(x)
        if reduce:
            x = self.reduction(x.unsqueeze(1)).squeeze(1)
        # x = self.dropout(x)

        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x

class arch33(nn.Module):

    def __init__(self, num_kernels=2): # first run had num_kernels=1, second run enabled bias weights, third has num_kernels=2
        super(arch33, self).__init__()
        self.num_kernels = num_kernels
        sizes = [4000, 2000, 1000, 500, 250]
        kernels = []
        for k in range(len(sizes)):
            for _ in range(num_kernels):
                kernels.append(KernelModule0(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.PReLU()
        # self.norm = nn.BatchNorm1d(num_features=125)
        # self.norm = nn.InstanceNorm1d(num_features=125)  # When batch=1
        # self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(125 * 125, 125)
        self.fc2 = nn.Linear(125, 5)

    def forward(self, x):
        for i, kernel in enumerate(self.kernels):
            if i < len(self.kernels) - 1:
                if i % self.num_kernels == (self.num_kernels - 1):
                    x = kernel(x, reduce=True)
                else:
                    x = kernel(x)
            else:
                x = kernel(x, fin=True, reduce=True)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = F.normalize(x, p=2, dim=1)
        # x = self.activation(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class arch333(nn.Module):

    def __init__(self, num_kernels=7):
        super(arch333, self).__init__()
        self.num_kernels = num_kernels
        sizes = [10000, 5000, 2500, 1250, 625, 312, 156]
        kernels = []
        for k in range(num_kernels):
            kernels.append(KernelModule(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(78 * 78, 125)
        self.fc2 = nn.Linear(125, 5)

    def forward(self, x):
        for i, kernel in enumerate(self.kernels):
            if i < self.num_kernels-1:
                x = kernel(x)
            else:
                x = kernel(x, fin=True)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class arch33_10000(nn.Module):  # this is the fourth run

    def __init__(self, num_kernels=1): 
        super(arch33_10000, self).__init__()
        self.num_kernels = num_kernels
        sizes = [10000, 5000, 2500, 1250, 625, 312, 156]
        kernels = []
        for k in range(len(sizes)):
            for _ in range(num_kernels):
                kernels.append(KernelModule0(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.PReLU()
        # self.norm = nn.BatchNorm1d(num_features=125)
        # self.norm = nn.InstanceNorm1d(num_features=125)  # When batch=1
        # self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(78 * 78, 125)
        self.fc2 = nn.Linear(125, 5)

    def forward(self, x):
        for i, kernel in enumerate(self.kernels):
            if i < len(self.kernels) - 1:
                if i % self.num_kernels == (self.num_kernels - 1):
                    x = kernel(x, reduce=True)
                else:
                    x = kernel(x)
            else:
                x = kernel(x, fin=True, reduce=True)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = F.normalize(x, p=2, dim=1)
        # x = self.activation(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x