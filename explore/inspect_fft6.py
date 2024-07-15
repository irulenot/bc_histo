import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image from skimage and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Zero out everything but the center 300 pixels
center_size = 100
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols), dtype=np.bool_)
mask[crow-center_size//2:crow+center_size//2, ccol-center_size//2:ccol+center_size//2] = True

# Compute the 2D FFT of the image
fft_image = np.fft.fft2(image)
fft_image[~mask] = 0
reconstructed_fft_image = np.fft.ifft2(fft_image)
reconstructed_fft_image = np.abs(reconstructed_fft_image)

# Shift the zero frequency component to the center of the spectrum
shifted_fft = np.fft.fftshift(np.fft.fft2(image))
shifted_fft[~mask] = 0
reconstructed_shifted_image = np.fft.ifft2(np.fft.ifftshift(shifted_fft))
reconstructed_shifted_image = np.abs(reconstructed_shifted_image)

# Create a figure with three subplots (original, FFT only, FFT + shift)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original grayscale image
axes[0].imshow(image, cmap='gray')
axes[0].axis('off')
axes[0].set_title(f'Original Image, size={image.shape}')

# Plot the first reconstructed image (FFT only)
axes[1].imshow(reconstructed_fft_image, cmap='gray')
axes[1].axis('off')
axes[1].set_title(f'FFT, crop_size={center_size}')

# Plot the second reconstructed image (FFT and shift)
axes[2].imshow(reconstructed_shifted_image, cmap='gray')
axes[2].axis('off')
axes[2].set_title(f'FFT and shift, crop_size={center_size}')

# Adjust layout
plt.tight_layout()

# Save the figure as a single PNG file
plt.savefig('img4.png', bbox_inches='tight', pad_inches=0)