import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# Load an image
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256))  # Resize to a smaller size for this example

# Perform 2-level DWT
coeffs = pywt.wavedec2(image, 'db1', level=2)

# Extract coefficients
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

# Plot the original image and the wavelet coefficients
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
fig.tight_layout()

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Level 1 coefficients
axes[0, 1].imshow(cA2, cmap='gray')
axes[0, 1].set_title('Approximation Level 2')
axes[0, 1].axis('off')

axes[0, 2].imshow(cH2, cmap='gray')
axes[0, 2].set_title('Horizontal Detail Level 2')
axes[0, 2].axis('off')

axes[0, 3].imshow(cV2, cmap='gray')
axes[0, 3].set_title('Vertical Detail Level 2')
axes[0, 3].axis('off')

axes[1, 0].imshow(cD2, cmap='gray')
axes[1, 0].set_title('Diagonal Detail Level 2')
axes[1, 0].axis('off')

# Level 2 coefficients
axes[1, 1].imshow(coeffs[0], cmap='gray')
axes[1, 1].set_title('Approximation Level 1')
axes[1, 1].axis('off')

axes[1, 2].imshow(cH1, cmap='gray')
axes[1, 2].set_title('Horizontal Detail Level 1')
axes[1, 2].axis('off')

axes[1, 3].imshow(cV1, cmap='gray')
axes[1, 3].set_title('Vertical Detail Level 1')
axes[1, 3].axis('off')

axes[2, 0].imshow(cD1, cmap='gray')
axes[2, 0].set_title('Diagonal Detail Level 1')
axes[2, 0].axis('off')

# Hide empty subplots
for i in range(1, 4):
    axes[2, i].axis('off')

plt.savefig('explore/rfft3.png')
