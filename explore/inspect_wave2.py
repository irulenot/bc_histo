import numpy as np
import pywt
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from pywt.data import camera

# Load example image
image = camera()

# Fourier Transform
def fourier_transform(img):
    return fft.fftshift(fft.fft2(img))

def inverse_fourier_transform(f_img):
    return np.abs(fft.ifft2(fft.ifftshift(f_img)))

# Wavelet Transform
def wavelet_transform(img, level=3):
    return pywt.wavedec2(img, 'haar', level=level)

def inverse_wavelet_transform(coeffs):
    return pywt.waverec2(coeffs, 'haar')

# Compute Fourier Transform
f_transform = fourier_transform(image)
f_reconstructed = inverse_fourier_transform(f_transform)

# Compute Wavelet Transform
wavelet_coeffs = wavelet_transform(image, level=3)
wavelet_reconstructed = inverse_wavelet_transform(wavelet_coeffs)

# Plotting
plt.figure(figsize=(18, 12))

# Original Image
plt.subplot(3, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Fourier Transform Magnitude
plt.subplot(3, 4, 2)
plt.title('Fourier Transform Magnitude')
plt.imshow(np.log(np.abs(f_transform) + 1), cmap='gray')
plt.axis('off')

# Fourier Transform Reconstruction
plt.subplot(3, 4, 3)
plt.title('Fourier Reconstruction')
plt.imshow(f_reconstructed, cmap='gray')
plt.axis('off')

# Wavelet Transform - Approximation Level 3
plt.subplot(3, 4, 5)
plt.title('Wavelet Level 3 - Approximation')
plt.imshow(wavelet_coeffs[0], cmap='gray')
plt.axis('off')

# Wavelet Transform - Details Level 3
plt.subplot(3, 4, 6)
plt.title('Wavelet Level 3 - Details')
details_level_3 = np.concatenate([
    np.concatenate(wavelet_coeffs[1], axis=1)  # Concatenate H, V, D details horizontally
], axis=0)  # Concatenate vertically with approximation
plt.imshow(details_level_3, cmap='gray')
plt.axis('off')

# Wavelet Transform - Approximation Level 2
plt.subplot(3, 4, 7)
plt.title('Wavelet Level 2 - Approximation')
plt.imshow(wavelet_coeffs[1][0], cmap='gray')
plt.axis('off')

# Wavelet Transform - Details Level 2
plt.subplot(3, 4, 8)
plt.title('Wavelet Level 2 - Details')
details_level_2 = np.concatenate([
    np.concatenate(wavelet_coeffs[2], axis=1)  # Concatenate H, V, D details horizontally
], axis=0)  # Concatenate vertically with approximation
plt.imshow(details_level_2, cmap='gray')
plt.axis('off')

# Wavelet Transform - Approximation Level 1
plt.subplot(3, 4, 9)
plt.title('Wavelet Level 1 - Approximation')
plt.imshow(wavelet_coeffs[2][0], cmap='gray')
plt.axis('off')

# Wavelet Transform - Details Level 1
plt.subplot(3, 4, 10)
plt.title('Wavelet Level 1 - Details')
details_level_1 = np.concatenate([
    np.concatenate(wavelet_coeffs[3], axis=1)  # Concatenate H, V, D details horizontally
], axis=0)  # Concatenate vertically with approximation
plt.imshow(details_level_1, cmap='gray')
plt.axis('off')

# Wavelet Reconstruction
plt.subplot(3, 4, 11)
plt.title('Wavelet Reconstruction')
plt.imshow(wavelet_reconstructed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('explore/wave2.png')
