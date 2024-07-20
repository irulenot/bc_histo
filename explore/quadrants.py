import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data
from itertools import combinations

# Load the astronaut image and convert to grayscale
image = color.rgb2gray(data.astronaut())

# Compute the 2D Fourier transform
fft_image = np.fft.fftshift(np.fft.fft2(image))

# Get the image shape
h, w = image.shape
half_h, half_w = h // 2, w // 2

# Define the quadrants
quadrants = {
    'Q1': (slice(0, half_h), slice(0, half_w)),
    'Q2': (slice(0, half_h), slice(half_w, w)),
    'Q3': (slice(half_h, h), slice(0, half_w)),
    'Q4': (slice(half_h, h), slice(half_w, w)),
}

# Function to mask all quadrants except for the given ones
def mask_quadrants(fft_image, quads):
    masked_fft = np.zeros_like(fft_image, dtype=complex)
    for quad in quads:
        q_slice = quadrants[quad]
        masked_fft[q_slice] = fft_image[q_slice]
    return masked_fft

# Create all combinations of two quadrants
quad_combinations = list(combinations(quadrants.keys(), 2))

# Prepare the plot
fig, axes = plt.subplots(2, len(quad_combinations) // 2, figsize=(15, 6))

for ax, quads in zip(axes.flatten(), quad_combinations):
    # Mask the FFT image
    masked_fft = mask_quadrants(fft_image, quads)
    
    # Inverse FFT to reconstruct the image
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(masked_fft))
    reconstructed_image = np.abs(reconstructed_image)
    
    # Display the reconstructed image
    ax.imshow(reconstructed_image, cmap='gray')
    ax.set_title(f'Quadrants: {quads}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('img.png')
