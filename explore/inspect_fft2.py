import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image from skimage and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Perform the 2D Fourier Transform
F = np.fft.fft2(image)

# Get the magnitude and phase spectra
magnitude = np.abs(F)
phase = np.angle(F)

# Get the dimensions of the image
rows, cols = F.shape

# Remove the symmetric part of the magnitude spectrum by zeroing out the appropriate elements
magnitude_modified = magnitude.copy()
for i in range(rows):
    for j in range(cols):
        if i > rows // 2 or (i == rows // 2 and j >= cols // 2):
            magnitude_modified[i, j] = 0

# Reconstruct the Fourier transform with the modified magnitude and the original phase
F_modified = magnitude_modified * np.exp(1j * phase)

# Perform the inverse Fourier Transform to reconstruct the image
reconstructed_image = np.fft.ifft2(F_modified)

# Compute magnitude and phase for visualization purposes
magnitude_original = np.abs(F)
magnitude_modified_visual = np.abs(F_modified)

# Create a figure to display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Magnitude spectrum (original)
axes[0, 1].imshow(np.log1p(magnitude_original), cmap='gray')
axes[0, 1].set_title('Magnitude Spectrum (Original)')
axes[0, 1].axis('off')

# Magnitude spectrum (modified)
axes[0, 2].imshow(np.log1p(magnitude_modified_visual), cmap='gray')
axes[0, 2].set_title('Magnitude Spectrum (Modified)')
axes[0, 2].axis('off')

# Reconstructed image (magnitude)
axes[1, 0].imshow(np.abs(reconstructed_image), cmap='gray')
axes[1, 0].set_title('Reconstructed Image (Magnitude)')
axes[1, 0].axis('off')

# Reconstructed image (real part)
axes[1, 1].imshow(np.real(reconstructed_image), cmap='gray')
axes[1, 1].set_title('Reconstructed Image (Real Part)')
axes[1, 1].axis('off')

# Reconstructed image (imaginary part)
axes[1, 2].imshow(np.imag(reconstructed_image), cmap='gray')
axes[1, 2].set_title('Reconstructed Image (Imaginary Part)')
axes[1, 2].axis('off')

# Display the plots
plt.tight_layout()
plt.savefig('img2.png')
