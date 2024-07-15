import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image from skimage and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Compute the 2D Fourier transform of the image
F = np.fft.fft2(image)
# Shift the zero-frequency component to the center of the spectrum
Fshift = np.fft.fftshift(F)
# Compute the phase spectrum
phase_spectrum = np.angle(Fshift)

# Use a uniform magnitude (e.g., magnitude of 1 for all frequencies)
uniform_magnitude = np.ones_like(Fshift)

# Reconstruct the Fourier transform with the uniform magnitude and original phase
F_reconstructed = uniform_magnitude * np.exp(1j * phase_spectrum)
# Inverse shift the zero-frequency component to the original position
F_ishift = np.fft.ifftshift(F_reconstructed)
# Compute the inverse 2D Fourier transform to get the reconstructed image
image_reconstructed = np.fft.ifft2(F_ishift)
# Take the real part of the inverse Fourier transform
image_reconstructed = np.real(image_reconstructed)

# Plot the original and reconstructed images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(image_reconstructed, cmap='gray')
axes[1].set_title('Reconstructed Image (Only Phase)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('img3.png', bbox_inches='tight', pad_inches=0)
