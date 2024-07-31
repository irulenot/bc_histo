import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image from skimage and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Calculate Fourier Transform
f = np.fft.fft2(original_image)
fshift = np.fft.fftshift(f)

# Calculate magnitude and phase
magnitude_spectrum = np.abs(fshift)
phase_spectrum = np.angle(fshift)

# Reconstruct images from magnitude and phase
f_magnitude = np.abs(f)  # Magnitude spectrum
f_phase = np.angle(f)    # Phase spectrum

# Inverse Fourier Transform to reconstruct images
img_from_magnitude = np.fft.ifft2(f_magnitude).real
img_from_phase = np.fft.ifft2(np.abs(f) * np.exp(1j * f_phase)).real

# Clip values to valid range (0-255)
img_from_magnitude = np.clip(img_from_magnitude, 0, 255).astype(np.uint8)
img_from_phase = np.clip(img_from_phase, 0, 255).astype(np.uint8)

# Plotting
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Magnitude Spectrum
plt.subplot(1, 4, 2)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# Phase Spectrum
plt.subplot(1, 4, 3)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum')
plt.axis('off')

# Image reconstructed from Magnitude
plt.subplot(1, 4, 4)
plt.imshow(img_from_magnitude, cmap='gray')
plt.title('Reconstructed from Magnitude')
plt.axis('off')

# Adjust layout for better display
plt.tight_layout()

# Save the figure
plt.savefig('image_processing_result.png')