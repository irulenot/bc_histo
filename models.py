import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import math

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

#####################################

class KernelModule3(nn.Module):
    def __init__(self, size=2000):
        super(KernelModule3, self).__init__()
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
        
        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        if reduce:
            x = self.reduction(x.unsqueeze(1)).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x
    
class arch3(nn.Module):

    def __init__(self, num_kernels=1): # BEST, 0.5066, acc=0.5
        super(arch3, self).__init__()
        self.num_kernels = num_kernels
        sizes = [4000, 2000, 1000, 500, 250]
        kernels = []
        for k in range(len(sizes)):
            for _ in range(num_kernels):
                kernels.append(KernelModule3(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.PReLU()

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
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class KernelModule31(nn.Module):
    def __init__(self, in_size=4000, out_size=3000):
        super(KernelModule31, self).__init__()
        self.magnitude_kernel = nn.Parameter(torch.randn(1, in_size, in_size))
        self.phase_kernel = nn.Parameter(torch.randn(1, in_size, in_size))
        self.out_size=out_size

    def forward(self, x, fin=False, reduce=False):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        magnitude = magnitude * self.magnitude_kernel
        phase = phase * self.phase_kernel
        
        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        if reduce:
            x = F.interpolate(x.unsqueeze(1), size=(self.out_size, self.out_size), mode='bicubic', align_corners=False).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x

class arch31(nn.Module):

    def __init__(self, num_kernels=1): # BEST, 0.5066, acc=0.5
        super(arch31, self).__init__()
        self.num_kernels = num_kernels
        in_sizes = [4000, 3000, 2000, 1000, 500, 250]
        out_sizes = [3000, 2000, 1000, 500, 250, 125]
        kernels = []
        for k in range(len(in_sizes)):
            for _ in range(num_kernels):
                kernels.append(KernelModule31(in_size=in_sizes[k], out_size=out_sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.PReLU()

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
        x = self.fc1(x)
        x = self.fc2(x)
        return x

####################################################

class arch4(nn.Module):

    def __init__(self):
        super(arch4, self).__init__()
        # self.num_kernels = num_kernels
        # self.kernel = KernelModule0(4000)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.PReLU()

        self.fc5 = nn.Linear(250 * 250, 128)
        self.bn5 = nn.LayerNorm(128)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(128, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.real.unsqueeze(1)
        x = self.dropout(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu4(self.bn4(self.conv4(x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)

class KernelModule41(nn.Module):
    def __init__(self, size=4000):
        super(KernelModule41, self).__init__()
        self.magnitude_kernel1 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel1 = nn.Parameter(torch.randn(1, size, size))

        self.magnitude_kernel2 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel2 = nn.Parameter(torch.randn(1, size, size))

        self.magnitude_kernel3 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel3 = nn.Parameter(torch.randn(1, size, size))

        self.mb = nn.Parameter(torch.randn(1, size, size))
        self.pb = nn.Parameter(torch.randn(1, size, size))

    def forward(self, x):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        magnitude = magnitude * self.magnitude_kernel1
        phase = phase * self.phase_kernel1

        magnitude = magnitude * self.magnitude_kernel2
        phase = phase * self.phase_kernel2

        magnitude = magnitude * self.magnitude_kernel3
        phase = phase * self.phase_kernel3

        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        return x

class arch41(nn.Module): # BEST  0.4048, acc 0.3125

    def __init__(self):
        super(arch41, self).__init__()
        # self.num_kernels = num_kernels
        self.kernel = KernelModule41(4000)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.PReLU()

        self.fc5 = nn.Linear(250 * 250, 128)
        self.bn5 = nn.LayerNorm(128)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(128, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        
        x = self.kernel(x).unsqueeze(1)

        x = self.dropout(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu4(self.bn4(self.conv4(x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)

class KernelModule42(nn.Module):
    def __init__(self, size=4000):
        super(KernelModule42, self).__init__()
        self.magnitude_kernel1 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel1 = nn.Parameter(torch.randn(1, size, size))

        self.magnitude_kernel2 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel2 = nn.Parameter(torch.randn(1, size, size))

        self.magnitude_kernel3 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel3 = nn.Parameter(torch.randn(1, size, size))

        self.mb = nn.Parameter(torch.randn(1, size, size))
        self.pb = nn.Parameter(torch.randn(1, size, size))

    def forward(self, x):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        m1 = magnitude * self.magnitude_kernel1
        p1 = phase * self.phase_kernel1

        m2 = magnitude * self.magnitude_kernel2
        p2 = phase * self.phase_kernel2

        m3 = magnitude * self.magnitude_kernel3
        p3 = phase * self.phase_kernel3

        magnitude = m1 + m2 + m3 + self.mb
        phase = p1 + p2 + p3 + self.pb

        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        return x

class KernelModule43(nn.Module):
    def __init__(self, size=4000):
        super(KernelModule43, self).__init__()
        self.magnitude_kernel = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel = nn.Parameter(torch.randn(1, size, size))

    def forward(self, x):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        magnitude = magnitude * self.magnitude_kernel
        phase = phase * self.phase_kernel

        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        return x

class arch42(nn.Module): # BEST  0.4048, acc 0.3125

    def __init__(self):
        super(arch42, self).__init__()
        # self.num_kernels = num_kernels
        self.kernel = KernelModule42(4000)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.PReLU()

        self.fc5 = nn.Linear(250 * 250, 128)
        self.bn5 = nn.LayerNorm(128)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(128, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        
        x = self.kernel(x).unsqueeze(1)

        x = self.dropout(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu4(self.bn4(self.conv4(x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)
    
class arch43(nn.Module):  # BEST = 0.3575

    def __init__(self):
        super(arch43, self).__init__()
        # self.num_kernels = num_kernels

        self.kernel1 = KernelModule41(4000)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.PReLU()
        self.kernel2 = KernelModule41(2000)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.PReLU()
        self.kernel3 = KernelModule41(1000)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.PReLU()
        self.kernel4 = KernelModule41(500)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.PReLU()

        self.fc5 = nn.Linear(250 * 250, 128)
        self.bn5 = nn.LayerNorm(128)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(128, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        
        x = self.kernel1(x).unsqueeze(1)
        x = self.dropout(self.relu1(self.bn1(self.conv1(x))))
        x = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1), dim=(-2, -1)), dim=(-2, -1))
        x = self.kernel2(x).unsqueeze(1)
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1), dim=(-2, -1)), dim=(-2, -1))
        x = self.kernel3(x).unsqueeze(1)
        x = self.dropout(self.relu3(self.bn3(self.conv3(x))))
        x = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1), dim=(-2, -1)), dim=(-2, -1))
        x = self.kernel4(x).unsqueeze(1)
        x = self.dropout(self.relu4(self.bn4(self.conv4(x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)
    
######################################################

class arch5(nn.Module):

    def __init__(self):
        super(arch5, self).__init__()

        # Define convolutional layers
        self.num_convs = 7  # Number of convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        in_channels = 1  # Initial input channels (grayscale image)

        for i in range(self.num_convs):
            out_channels = 1  # Assuming 1 output channel for each convolutional layer
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
            bn_layer = nn.BatchNorm2d(out_channels)
            relu_layer = nn.PReLU()
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
            self.relu_layers.append(relu_layer)

        self.fc5 = nn.Linear(156 * 156, 156)
        self.bn5 = nn.LayerNorm(156)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(156, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.real.unsqueeze(1)
        for i in range(self.num_convs):
            x = self.dropout(self.relu_layers[i](self.bn_layers[i](self.conv_layers[i](x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)
    
class KernelModule51(nn.Module):
    def __init__(self, size=20000):
        super(KernelModule51, self).__init__()
        self.magnitude_kernel1 = nn.Parameter(torch.randn(1, size, size))
        self.phase_kernel1 = nn.Parameter(torch.randn(1, size, size))

    def forward(self, x):
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        magnitude = magnitude * self.magnitude_kernel1
        phase = phase * self.phase_kernel1

        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined.cpu(), dim=(-2, -1)), dim=(-2, -1)).real
        return x.to(combined.device)

class arch51(nn.Module):

    def __init__(self):
        super(arch51, self).__init__()

        self.kernel = KernelModule51(20000)

        # Define convolutional layers
        self.num_convs = 5  # Number of convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        in_channels = 1  # Initial input channels (grayscale image)

        for i in range(self.num_convs):
            out_channels = 1  # Assuming 1 output channel for each convolutional layer
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3)
            bn_layer = nn.BatchNorm2d(out_channels)
            relu_layer = nn.PReLU()
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
            self.relu_layers.append(relu_layer)

        self.fc5 = nn.Linear(82 * 82, 82)
        self.bn5 = nn.LayerNorm(82)
        self.relu5 = nn.PReLU()
        self.fc6 = nn.Linear(82, 5)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(1)
        for i in range(self.num_convs):
            x = self.dropout(self.relu_layers[i](self.bn_layers[i](self.conv_layers[i](x))))

        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu5(self.bn5(self.fc5(x))))
        return self.fc6(x)
    
class KernelModule52(nn.Module):
    def __init__(self, size=2000):
        super(KernelModule52, self).__init__()
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
        
        combined = magnitude * torch.exp(1j * phase)
        x = torch.fft.ifft2(torch.fft.ifftshift(combined, dim=(-2, -1)), dim=(-2, -1)).real
        if reduce:
            x = self.reduction(x.unsqueeze(1)).squeeze(1)
        if fin == False:
            x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return x
    
class arch52(nn.Module):

    def __init__(self, num_kernels=1): # BEST, 0.5066, acc=0.5
        super(arch52, self).__init__()
        self.num_kernels = num_kernels
        sizes = [20000, 10000, 5000, 1000, 500, 250]
        kernels = []
        for k in range(len(sizes)):
            for _ in range(num_kernels):
                kernels.append(KernelModule3(size=sizes[k]))
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.PReLU()

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
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
####################################################

class arch6(nn.Module):

    def __init__(self):
        super(arch6, self).__init__()

        self.magnitude_proj = nn.Linear(19500, 1950)
        self.magnitude_proj_norm = nn.LayerNorm(1950)

        self.magnitude_pos_embed = nn.Parameter(torch.randn(1, 2000, 1950))

        magnitude_layer = nn.TransformerEncoderLayer(d_model=1950, nhead=5)
        self.magnitude_encoder = nn.TransformerEncoder(magnitude_layer, num_layers=3)

        self.classification_head = nn.Sequential(
            nn.Linear(2000 * 1950, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, TB, LR):
        TB, LR = TB.unsqueeze(0), LR.unsqueeze(0)
        TB_magnitude = TB.real[..., :-500]
        LR_magnitude = LR.real[..., :-500]

        magnitude = torch.cat([TB_magnitude, LR_magnitude], dim=1)
        magnitude = magnitude.reshape(magnitude.size(0), 2000, -1)

        magnitude = self.magnitude_proj_norm(self.magnitude_proj(magnitude[0]))
        magnitude = magnitude + self.magnitude_pos_embed

        magnitude = self.magnitude_encoder(magnitude)

        magnitude = magnitude.view(1, -1)
        output = self.classification_head(magnitude)

        return output

class arch61(nn.Module):

    def __init__(self):
        super(arch61, self).__init__()

        self.magnitude_proj = nn.Linear(19500, 1950)

        magnitude_layer = nn.TransformerEncoderLayer(d_model=1950, nhead=5)
        self.magnitude_encoder = nn.TransformerEncoder(magnitude_layer, num_layers=3)

        self.classification_head = nn.Sequential(
            nn.Linear(2000 * 1950, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, TB, LR):
        TB, LR = TB.unsqueeze(0), LR.unsqueeze(0)
        TB_magnitude = TB.real[..., :-500]
        LR_magnitude = LR.real[..., :-500]

        magnitude = torch.cat([TB_magnitude, LR_magnitude], dim=1)
        magnitude = magnitude.reshape(magnitude.size(0), 2000, -1)

        magnitude = self.magnitude_proj(magnitude[0])

        magnitude = self.magnitude_encoder(magnitude)

        magnitude = magnitude.view(1, -1)
        output = self.classification_head(magnitude)

        return output
    
class arch65(nn.Module):

    def __init__(self):
        super(arch65, self).__init__()

        self.proj_T = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
        )
        self.proj_B = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
        )
        self.proj_L = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
        )
        self.proj_R = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
        )

        self.pos_embed = nn.Parameter(torch.randn(4, 250, 9750))
        self.initial = nn.Parameter(torch.randn(9750, 1, 1000))

        layer = nn.TransformerDecoderLayer(d_model=1000, nhead=5, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(37088, 3708),
            nn.LayerNorm(3708),
            nn.GELU(),
            nn.Linear(3708, 370),
            nn.LayerNorm(370),
            nn.GELU(),
            nn.Linear(370, 37),
            nn.LayerNorm(37),
            nn.GELU(),
            nn.Linear(37, 5)
        )

    def forward(self, TB, LR):
        T, B, L, R = TB[0][:, :-500].real, TB[1][:, :-500].real, LR[0][:, :-500].real, LR[1][:, :-500].real

        # Do conv instead of linear projections
        # Try normalizing data before input
        # Attention-based convolutions?
        # Need phase info?
        # Try using multiplication rather than conv for projection?
        # Use conv to get representations but not transformer for fusing
        # Maybe transposing the LR is what is causing the issues?

        T = self.proj_T(T.unsqueeze(0).unsqueeze(0)).squeeze()
        B = self.proj_B(B.unsqueeze(0).unsqueeze(0)).squeeze()
        L = self.proj_L(L.unsqueeze(0).unsqueeze(0)).squeeze()
        R = self.proj_R(R.unsqueeze(0).unsqueeze(0)).squeeze()

        x = torch.stack([T, B, L, R])
        x += self.pos_embed
        
        x = x.reshape(1000, 1, 9750)
        x = x.permute(2, 1, 0)
        x = self.decoder(x, self.initial)
        x = x.permute(1, 0, 2)

        x = self.conv(x.unsqueeze(1))
        x = x.view(1, -1)
        return self.classification_head(x)
    
class arch66(nn.Module):

    def __init__(self):
        super(arch66, self).__init__()

        self.proj_T = nn.Sequential(
            nn.LayerNorm(19500),
            nn.Linear(19500, 1950),
            nn.LayerNorm(1950),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1950, 195),
            nn.LayerNorm(195),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(195, 19),
        )
        self.proj_B = nn.Sequential(
            nn.LayerNorm(19500),
            nn.Linear(19500, 1950),
            nn.LayerNorm(1950),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1950, 195),
            nn.LayerNorm(195),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(195, 19),
        )
        self.proj_L = nn.Sequential(
            nn.LayerNorm(19500),
            nn.Linear(19500, 1950),
            nn.LayerNorm(1950),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1950, 195),
            nn.LayerNorm(195),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(195, 19),
        )
        self.proj_R = nn.Sequential(
            nn.LayerNorm(19500),
            nn.Linear(19500, 1950),
            nn.LayerNorm(1950),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1950, 195),
            nn.LayerNorm(195),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(195, 19),
        )

        self.pos_embed = nn.Parameter(torch.randn(4, 500, 19))
        self.initial = nn.Parameter(torch.randn(50, 1, 760))

        layer = nn.TransformerDecoderLayer(d_model=760, nhead=5)
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)

        self.classification_head = nn.Sequential(
            nn.Linear(38000, 3800),
            nn.LayerNorm(3800),
            nn.GELU(),
            nn.Linear(3800, 380),
            nn.LayerNorm(380),
            nn.GELU(),
            nn.Linear(380, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Linear(38, 5)
        )

    def forward(self, TB, LR):
        T, B, L, R = TB[0][:, :-500].real, TB[1][:, :-500].real, LR[0][:, :-500].real, LR[1][:, :-500].real

        # Is the issue the convolutions at the end? IS the classifcation head the issue?
        # Should I be doing normalization before my linear layer for FFT data?

        T = self.proj_T(T)
        B = self.proj_B(B)
        L = self.proj_L(L)
        R = self.proj_R(R)

        x = torch.stack([T, B, L, R])
        x += self.pos_embed
        
        x = x.permute(1, 0, 2).reshape(50, -1).unsqueeze(1)
        x = self.decoder(x, self.initial)

        x = x.flatten().unsqueeze(0)
        return self.classification_head(x)
    
class arch67(nn.Module):

    def __init__(self):
        super(arch67, self).__init__()

        mesh_path = '/data/breast-cancer/PANDA/train_images_FFT_WSI_grayscaled/mesh.npz'
        mesh = np.load(mesh_path)
        TM, BM, LM, RM = mesh['array1'][0], mesh['array1'][1], mesh['array2'][0], mesh['array2'][1]
        self.mesh = nn.Parameter(torch.cat([torch.tensor(TM), torch.tensor(BM), torch.tensor(LM), torch.tensor(RM)], dim=0).to(torch.float32))
        self.pos_embed = nn.Parameter(torch.randn(self.mesh.shape))

        self.norm_T = nn.LayerNorm(20000)
        self.norm_B = nn.LayerNorm(20000)
        self.norm_L = nn.LayerNorm(20000)
        self.norm_R = nn.LayerNorm(20000)
        self.norm = nn.LayerNorm(20000)
        self.mul = nn.Parameter(torch.randn(20000))
        self.proj = nn.Linear(20000, 10000)
        self.proj_norm = nn.LayerNorm(10000)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, TB, LR):
        T, B, L, R = TB[0].real, TB[1].real, LR[0].real, LR[1].real

        T, B, L, R = self.norm_T(T), self.norm_B(B), self.norm_L(L), self.norm_R(R)
        x = torch.cat([T, B, L, R], dim=0)
        x = self.mul * self.norm(x)
        x = x + torch.sum(self.mesh * self.pos_embed, dim=2)
        x = x.reshape(4, 500, 20000)

        x = self.conv(x.unsqueeze(0))
        
        x = x.view(1, -1)
        return self.classification_head(x)
    
class arch68(nn.Module):

    def __init__(self):
        super(arch68, self).__init__()

        self.conv_T = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_B = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_L = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_R = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )

        self.pos_emb = nn.Parameter(torch.randn(160, 128))

        magnitude_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.encoder = nn.TransformerEncoder(magnitude_layer, num_layers=3)

        self.classification_head = nn.Sequential(
            nn.Linear(20480, 10240),
            nn.LayerNorm(10240),
            nn.ReLU(),
            nn.Linear(10240, 5120),
            nn.LayerNorm(5120),
            nn.ReLU(),
            nn.Linear(5120, 2560),
            nn.LayerNorm(2560),
            nn.ReLU(),
            nn.Linear(2560, 1280),
            nn.LayerNorm(1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.LayerNorm(640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def forward(self, TB, LR):
        T, B, L, R = TB[0].real, TB[1].real, LR[0].real, LR[1].real
        T, B, L, R = self.conv_T(T.unsqueeze(0).unsqueeze(0)), self.conv_B(B.unsqueeze(0).unsqueeze(0)), self.conv_L(L.unsqueeze(0).unsqueeze(0)), self.conv_R(R.unsqueeze(0).unsqueeze(0)) 

        x = torch.cat([T.squeeze().T, B.squeeze().T, L.squeeze().T, R.squeeze().T], dim=0)
        x += self.pos_emb
        x = x.unsqueeze(0)
        x = self.encoder(x)

        x = x.view(1, -1)
        output = self.classification_head(x)

        return output
    
class arch69(nn.Module):

    def __init__(self):
        super(arch69, self).__init__()

        self.conv_TM = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_BM = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_LM = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_RM = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_TP = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_BP = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_LP = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_RP = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=500, stride=500),
            nn.BatchNorm2d(128),
        )
        self.conv_T = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
        self.conv_B = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
        self.conv_L = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
        self.conv_R = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )

        self.pos_emb = nn.Parameter(torch.randn(160, 128))

        magnitude_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.encoder = nn.TransformerEncoder(magnitude_layer, num_layers=3)

        self.classification_head = nn.Sequential(
            nn.Linear(20480, 10240),
            nn.LayerNorm(10240),
            nn.ReLU(),
            nn.Linear(10240, 5120),
            nn.LayerNorm(5120),
            nn.ReLU(),
            nn.Linear(5120, 2560),
            nn.LayerNorm(2560),
            nn.ReLU(),
            nn.Linear(2560, 1280),
            nn.LayerNorm(1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.LayerNorm(640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def forward(self, TB, LR):
        TM, BM, LM, RM = torch.abs(TB[0]), torch.abs(TB[1]), torch.abs(LR[0]), torch.abs(LR[1])
        TP, BP, LP, RP = torch.angle(TB[0]), torch.angle(TB[1]), torch.angle(LR[0]), torch.angle(LR[1])
        TM, BM, LM, RM = self.conv_TM(TM.unsqueeze(0).unsqueeze(0)), self.conv_BM(BM.unsqueeze(0).unsqueeze(0)), self.conv_LM(LM.unsqueeze(0).unsqueeze(0)), self.conv_RM(RM.unsqueeze(0).unsqueeze(0)) 
        TP, BP, LP, RP = self.conv_TP(TP.unsqueeze(0).unsqueeze(0)), self.conv_BP(BP.unsqueeze(0).unsqueeze(0)), self.conv_LP(LP.unsqueeze(0).unsqueeze(0)), self.conv_RP(RP.unsqueeze(0).unsqueeze(0)) 

        T, B, L, R = torch.cat([TM, TP], dim=0).squeeze().unsqueeze(0), torch.cat([BM, BP], dim=0).squeeze().unsqueeze(0), torch.cat([RM, RP], dim=0).squeeze().unsqueeze(0), torch.cat([LM, LP], dim=0).squeeze().unsqueeze(0)
        T, B, L, R = self.conv_T(T), self.conv_B(B), self.conv_L(L), self.conv_R(R)
        x = torch.cat([T.squeeze().T, B.squeeze().T, L.squeeze().T, R.squeeze().T], dim=0)
        x += self.pos_emb
        x = x.unsqueeze(0)
        x = self.encoder(x)

        x = x.view(1, -1)
        output = self.classification_head(x)

        return output
    

#################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class arch7(nn.Module):

    def __init__(self):
        super(arch7, self).__init__()

        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1),
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
        )
        self.head = nn.Sequential(
            nn.Linear(10016, 1001),
            nn.LayerNorm(1001),
            nn.ReLU(),
            nn.Linear(1001, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, 5),
        )

    def forward(self, H, V):
        H, V = H.real, V.T.real
        x = torch.cat([H, V], dim=0)
        x = (x - x.mean()) / x.std()
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class arch71(nn.Module):

    def __init__(self):
        super(arch71, self).__init__()

        self.conv_H = nn.Sequential(
            nn.Conv1d(in_channels=20000, out_channels=5000, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(5000),
            nn.PReLU(),
            nn.Conv1d(in_channels=5000, out_channels=2500, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2500),
            nn.PReLU(),
            nn.Conv1d(in_channels=2500, out_channels=1250, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1250),
            nn.PReLU(),
            nn.Conv1d(in_channels=1250, out_channels=1000, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
        )
        self.conv_V = nn.Sequential(
            nn.Conv1d(in_channels=20000, out_channels=5000, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(5000),
            nn.PReLU(),
            nn.Conv1d(in_channels=5000, out_channels=2500, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2500),
            nn.PReLU(),
            nn.Conv1d(in_channels=2500, out_channels=1250, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1250),
            nn.PReLU(),
            nn.Conv1d(in_channels=1250, out_channels=1000, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(125**2, 64**2),
            nn.LayerNorm(64**2),
            nn.PReLU(),
            nn.Linear(64**2, 32**2),
            nn.LayerNorm(32**2),
            nn.PReLU(),
            nn.Linear(32**2, 16**2),
            nn.LayerNorm(16**2),
            nn.PReLU(),
            nn.Linear(16**2, 8**2),
            nn.LayerNorm(8**2),
            nn.PReLU(),
            nn.Linear(8**2, 4**2),
            nn.LayerNorm(4**2),
            nn.PReLU(),
            nn.Linear(4**2, 5),
        )

    def forward(self, H, V):
        H, V = H.T.real, V.real
        H, V = self.conv_H(H.unsqueeze(0)), self.conv_V(V.unsqueeze(0))
        x = torch.cat([H, V], dim=0).unsqueeze(0)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
    
class arch72(nn.Module):

    def __init__(self):
        super(arch72, self).__init__()

        self.conv_H = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
        )
        self.conv_V = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1000),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4977, 497),
            nn.LayerNorm(497),
            nn.PReLU(),
            nn.Linear(497, 49),
            nn.LayerNorm(49),
            nn.PReLU(),
            nn.Linear(49, 5),
        )

    def normalize(self, H, V):
        C = torch.cat((H.flatten(), V.flatten()), dim=0)
        H_normalized = (H - C.mean()) / C.std()
        V_normalized = (V - C.mean()) / C.std()
        return H_normalized, V_normalized

    def forward(self, H, V):
        H, V = H.real, V.T.real
        H, V = self.normalize(H, V)
        H, V = self.conv_H(H.unsqueeze(0)), self.conv_V(V.unsqueeze(0))
        x = torch.cat([H, V], dim=0).unsqueeze(0)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
    
class arch73(nn.Module):

    def __init__(self):
        super(arch73, self).__init__()

        self.proj_H = nn.Sequential(
            nn.Linear(62500, 6250),
            nn.LayerNorm(6250),
        )
        self.proj_V = nn.Sequential(
            nn.Linear(62500, 6250),
            nn.LayerNorm(6250),
        )
        self.pos_emb1 = nn.Parameter(torch.randn(640, 1))
        self.proj = nn.Sequential(
            nn.Linear(6250, 625),
            nn.LayerNorm(625),
        )
        self.pos_emb2 = nn.Parameter(torch.randn(640, 1))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(40**2, 16**2),
            nn.LayerNorm(16**2),
            nn.PReLU(),
            nn.Linear(16**2, 4**2),
            nn.LayerNorm(4**2),
            nn.PReLU(),
            nn.Linear(4**2, 5),
        )

    def normalize(self, H, V):
        C = torch.cat((H.flatten(), V.flatten()), dim=0)
        H_normalized = (H - C.mean()) / C.std()
        V_normalized = (V - C.mean()) / C.std()
        return H_normalized, V_normalized

    def forward(self, H, V):
        H, V = H.real, V.real
        H, V = self.normalize(H, V)
        H, V = H.reshape(-1, 250, 250), V.reshape(-1, 250, 250)
        H, V = self.proj_H(H.view(H.size(0), -1)), self.proj_V(H.view(H.size(0), -1))
        x = torch.cat([H, V], dim=0) + self.pos_emb1
        x = self.proj(x) + self.pos_emb2
        x = self.conv(x.unsqueeze(0).unsqueeze(0))
        x = x.view(x.size(0), -1)
        return self.head(x)