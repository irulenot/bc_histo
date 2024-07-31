import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch

image = torch.tensor(np.load('presentation/WSI_image.npz')['arr_0'])
transform = transforms.Grayscale(num_output_channels=1)
image = transform(image).squeeze().numpy()

# Perform FFT
fft_image = fft2(image)

# Shift the FFT
shifted_fft_image = fftshift(fft_image)

# Plot and save the images side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# FFT of the image
axes[1].imshow(np.log(np.abs(fft_image)))
axes[1].set_title('FFT')
axes[1].axis('off')

# Shifted FFT
axes[2].imshow(np.log(np.abs(shifted_fft_image)))
axes[2].set_title('Shifted FFT')
axes[2].axis('off')

# Save the plot
plt.savefig('presentation/fft.png')