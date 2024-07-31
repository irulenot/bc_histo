import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image from skimage and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Perform the 2D Fourier Transform
F = np.fft.fft2(image)

# Compute the magnitude and phase for visualization purposes
magnitude = np.abs(F)
phase = np.angle(F)

# Shift the zero-frequency component to the center of the spectrum
F_shifted = np.fft.fftshift(F)
magnitude_shifted = np.abs(F_shifted)
phase_shifted = np.angle(F_shifted)

# Create a figure to display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Magnitude spectrum (unshifted)
axes[0, 1].imshow(np.log1p(magnitude), cmap='gray')
axes[0, 1].set_title('Magnitude Spectrum (Unshifted)')
axes[0, 1].axis('off')

# Phase spectrum (unshifted)
axes[0, 2].imshow(phase, cmap='gray')
axes[0, 2].set_title('Phase Spectrum (Unshifted)')
axes[0, 2].axis('off')

# Original image
axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title('Original Image')
axes[1, 0].axis('off')

# Magnitude spectrum (shifted)
axes[1, 1].imshow(np.log1p(magnitude_shifted), cmap='gray')
axes[1, 1].set_title('Magnitude Spectrum (Shifted)')
axes[1, 1].axis('off')

# Phase spectrum (shifted)
axes[1, 2].imshow(phase_shifted, cmap='gray')
axes[1, 2].set_title('Phase Spectrum (Shifted)')
axes[1, 2].axis('off')

# Demonstrating Hermitian symmetry
# Let's show the real part of the Fourier transform and its complex conjugate symmetry
# real_part = np.real(F)
# axes[1, 2].imshow(real_part, cmap='gray')
# axes[1, 2].set_title('Real Part of Fourier Transform')
# axes[1, 2].axis('off')

# Display the plots
plt.tight_layout()
plt.savefig('img1.png')
