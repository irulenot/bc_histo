import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# Load the astronaut image and convert it to grayscale
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256))  # Resize to a smaller size for this example

# Convert the image to a PyTorch tensor
image_tensor = torch.tensor(image, dtype=torch.float32)

# Apply the 2D real-valued Fourier transform
frequency_domain = torch.fft.rfft2(image_tensor)

# Compute the frequency bins for both dimensions
height, width = image_tensor.shape
freqs_y = torch.fft.fftfreq(height, d=1.0)
freqs_x = torch.fft.rfftfreq(width, d=1.0)

# Compute the magnitudes of the FFT coefficients
magnitude_spectrum = torch.abs(frequency_domain)

# Visualize the frequency spectrum along the vertical center line
center_freq_spectrum = magnitude_spectrum[:, width//2].numpy()

# Plot the original image, magnitude spectrum, and center frequency spectrum
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Magnitude Spectrum")
plt.imshow(np.log(magnitude_spectrum.numpy() + 1), cmap='gray')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Center Frequency Spectrum")
plt.plot(freqs_y.numpy(), center_freq_spectrum)
plt.xlabel('Frequency (cycles per image)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.savefig('explore/rfft2.png')