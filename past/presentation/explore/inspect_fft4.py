import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import data, color

# Load the astronaut image and convert to grayscale
astronaut_rgb = data.astronaut()
astronaut_gray = color.rgb2gray(astronaut_rgb)

# Define dimensions for the grid
grid_size = (3, 3)
image_size = (128, 128)  # Size for each astronaut image in the grid

# Resize grayscale image to desired size
astronaut_gray_resized = transform.resize(astronaut_gray, image_size)

# Convert resized grayscale image to tensor and add batch dimension
original_image = transform.ToTensor()(astronaut_gray_resized).unsqueeze(0)

# Create a grid of the original image (9 repetitions in a 3x3 grid)
grid_image = torch.cat([original_image] * (grid_size[0] * grid_size[1]), dim=0)

# Compute FFT for both images
fft_original = torch.fft.fftn(original_image, dim=[-2, -1])
fft_grid = torch.fft.fftn(grid_image, dim=[-2, -1])

# Compute magnitude spectra (absolute value of FFT)
magnitude_original = torch.log(torch.abs(fft_original) + 1e-9)  # Log for visualization
magnitude_grid = torch.log(torch.abs(fft_grid) + 1e-9)

# Plotting
plt.figure(figsize=(18, 9))

# Plot the original grayscale image
plt.subplot(2, 2, 1)
plt.imshow(astronaut_gray, cmap='gray')
plt.title('Original Astronaut Image (Grayscale)')
plt.axis('off')

# Plot the resized grayscale image
plt.subplot(2, 2, 2)
plt.imshow(astronaut_gray_resized, cmap='gray')
plt.title('Resized Astronaut Image (Grayscale)')
plt.axis('off')

# Plot the FFT magnitude images
plt.subplot(2, 2, 3)
plt.imshow(magnitude_original.squeeze().cpu().numpy(), cmap='jet')  # Colored FFT magnitude
plt.title('FFT Magnitude - Single Astronaut Image')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(magnitude_grid.sum(dim=0).squeeze().cpu().numpy(), cmap='jet')  # Colored FFT magnitude
plt.title('FFT Magnitude - Grid Image')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.savefig('img.png')
