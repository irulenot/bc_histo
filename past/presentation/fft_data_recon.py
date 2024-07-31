import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch

image = torch.tensor(np.load('presentation/WSI_image.npz')['arr_0'])
transform = transforms.Grayscale(num_output_channels=1)
image = transform(image).squeeze()

crop_size = 3500
h, w = image.shape[0], image.shape[1]
start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
mask = torch.zeros(image.shape)
mask[start_h:start_h+crop_size, start_w:start_w+crop_size] = 1

transform = transforms.ToPILImage()
# Perform FFT
fft_image = torch.fft.fft2(image) * mask
fft_image = torch.fft.ifft2(fft_image).real
fft_image = np.array(transform(fft_image))

# Shift the FFT
shifted_fft_image = torch.fft.fftshift(torch.fft.fft2(image)) * mask
shifted_fft_image = torch.fft.ifft2(torch.fft.ifftshift(shifted_fft_image)).real
shifted_fft_image = np.array(transform(shifted_fft_image))

# Plot and save the images side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title(f'Original Image, size={tuple(image.shape)}')
axes[0].axis('off')
# FFT of the image
axes[1].imshow(fft_image, cmap='gray')
axes[1].set_title(f'FFT, crop_size={crop_size}')
axes[1].axis('off')
# Shifted FFT
axes[2].imshow(shifted_fft_image, cmap='gray')
axes[2].set_title(f'Shifted FFT, crop_size={crop_size}')
axes[2].axis('off')

# Save the plot
plt.savefig('presentation/fft_recon.png')