import torch
import torch.fft as fft
import torchvision.transforms.functional as TF
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

# Load the astronaut image from skimage
img = data.astronaut()

# Convert image to grayscale
img_gray = TF.to_grayscale(TF.to_pil_image(img))

# Convert PIL image to PyTorch tensor
img_tensor = TF.to_tensor(img_gray)

# Perform FFT
fft_img = fft.fft2(img_tensor, dim=(-2, -1))

# Reconstruct image from full FFT
reconstructed_img = fft.ifft2(fft_img, dim=(-2, -1))

# Take the real part of the reconstructed image (since we dealt with real data)
reconstructed_img = reconstructed_img.real

# Ensure values are in valid image range [0, 1]
reconstructed_img = torch.clamp(reconstructed_img, 0, 1)

# Convert back to PIL image for visualization
reconstructed_img_pil = TF.to_pil_image(reconstructed_img)

# Compute magnitude spectrum of original FFT
fft_mag = torch.abs(fft_img)

# Convert magnitude spectrum to numpy array for plotting
fft_mag_np = fft_mag.squeeze().numpy()

# Display original image, reconstructed image, and FFT magnitude
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_img_pil, cmap='gray')
plt.title('Reconstructed Image (Full FFT)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.log10(fft_mag_np), cmap='jet', vmin=np.log10(fft_mag_np.min()), vmax=np.log10(fft_mag_np.max()))
plt.colorbar(label='Log Magnitude')
plt.title('FFT Magnitude')
plt.axis('off')

plt.tight_layout()
plt.savefig('img.png')