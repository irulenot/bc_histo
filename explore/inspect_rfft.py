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

# Shift the zero-frequency component to the center
shifted_frequency_domain = torch.fft.fftshift(frequency_domain, dim=(-2, -1))

# Crop the middle part of the transformed image (e.g., central 128x128 region)
height, width = shifted_frequency_domain.shape
crop_size = 64
cropped_frequency_domain_shifted = shifted_frequency_domain[
    height//2 - crop_size//2:height//2 + crop_size//2,
    width//2 - crop_size//2:width//2 + crop_size//2
]

# Pad the cropped region back to the original size with zeros
padded_frequency_domain_shifted = torch.zeros_like(shifted_frequency_domain)
padded_frequency_domain_shifted[
    height//2 - crop_size//2:height//2 + crop_size//2,
    width//2 - crop_size//2:width//2 + crop_size//2
] = cropped_frequency_domain_shifted

# Shift the zero-frequency component back to the original position
unshifted_frequency_domain = torch.fft.ifftshift(padded_frequency_domain_shifted, dim=(-2, -1))

# Reconstruct the image by applying the inverse 2D Fourier transform
reconstructed_image_shifted = torch.fft.irfft2(unshifted_frequency_domain, s=image_tensor.shape)

# Crop the middle part of the original unshifted frequency domain
cropped_frequency_domain_unshifted = frequency_domain[
    height//2 - crop_size//2:height//2 + crop_size//2,
    :crop_size
]

# Pad the cropped region back to the original size with zeros
padded_frequency_domain_unshifted = torch.zeros_like(frequency_domain)
padded_frequency_domain_unshifted[
    height//2 - crop_size//2:height//2 + crop_size//2,
    :crop_size
] = cropped_frequency_domain_unshifted

# Reconstruct the image by applying the inverse 2D Fourier transform
reconstructed_image_unshifted = torch.fft.irfft2(padded_frequency_domain_unshifted, s=image_tensor.shape)

# Convert the reconstructed images to NumPy arrays for visualization
reconstructed_image_shifted_np = reconstructed_image_shifted.numpy()
reconstructed_image_unshifted_np = reconstructed_image_unshifted.numpy()

# Plot the original, reconstructed images and the rFFT data
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Reconstructed Image (Shifted)")
plt.imshow(reconstructed_image_shifted_np, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Reconstructed Image (Unshifted)")
plt.imshow(reconstructed_image_unshifted_np, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("rFFT Data (Shifted)")
plt.imshow(torch.log(torch.abs(shifted_frequency_domain) + 1).numpy(), cmap='gray')
plt.gca().add_patch(plt.Rectangle(
    (width//2 - crop_size//2, height//2 - crop_size//2), crop_size, crop_size,
    edgecolor='red', facecolor='none', linewidth=2
))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("rFFT Data (Unshifted)")
plt.imshow(torch.log(torch.abs(frequency_domain) + 1).numpy(), cmap='gray')
plt.gca().add_patch(plt.Rectangle(
    (width//2 - crop_size//2, height//2 - crop_size//2), crop_size, crop_size,
    edgecolor='red', facecolor='none', linewidth=2
))
plt.axis('off')

plt.tight_layout()
plt.savefig('explore/rfft.png')
plt.show()
