import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the image data
image = torch.tensor(np.load('presentation/WSI_image.npz')['arr_0'])

# Initialize lists to store magnitude and phase tensors
magnitude_list = []
phase_list = []
central_size = 1000

# Iterate over each channel
for c in range(3):
    # Extract the channel
    channel = image[c, ...]

    # Perform FFT
    fft_result = torch.fft.fft2(channel, dim=(-2, -1))

    # Compute magnitude and phase
    magnitude = torch.abs(fft_result)
    phase = torch.angle(fft_result)

    # Zero out pixels except central region
    h, w = magnitude.shape[-2], magnitude.shape[-1]
    h_start = (h - central_size) // 2
    w_start = (w - central_size) // 2

    magnitude[:h_start, :] = 0
    magnitude[h_start + central_size:, :] = 0
    magnitude[:, :w_start] = 0
    magnitude[:, w_start + central_size:] = 0

    phase[:h_start, :] = 0
    phase[h_start + central_size:, :] = 0
    phase[:, :w_start] = 0
    phase[:, w_start + central_size:] = 0

    # Append to lists
    magnitude_list.append(magnitude)
    phase_list.append(phase)

# Create a single figure for all magnitude and phase images
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot magnitude images
for c in range(3):
    axs[c, 0].imshow(np.log(magnitude_list[c].numpy()))
    axs[c, 0].set_title(f'Channel {c+1} Magnitude')
    axs[c, 0].axis('off')

# Plot phase images
for c in range(3):
    axs[c, 1].imshow(phase_list[c].numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axs[c, 1].set_title(f'Channel {c+1} Phase')
    axs[c, 1].axis('off')

fig.suptitle('FFT', fontsize=16, fontweight='bold')

# Adjust spacing and layout
fig.tight_layout(pad=0.0)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig('all_channels_combined3.png')
plt.close()

print("Magnitude and phase images saved together in a single plot with minimized white space.")
