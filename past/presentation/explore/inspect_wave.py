import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt

# Load example image
image = pywt.data.camera()

# Perform wavelet transform
coeffs = pywt.wavedec2(image, 'haar', level=2)

# Print the shapes of the coefficients for reference
print("Original Coefficients Shapes:")
for i, coeff in enumerate(coeffs):
    if isinstance(coeff, tuple):
        print(f"Level {i}: {tuple(c.shape for c in coeff)}")
    else:
        print(f"Approximation: {coeff.shape}")

# Crop the first level of detail coefficients
coeffs_cropped = list(coeffs)
coeffs_cropped[1] = (coeffs_cropped[1][0][::2, ::2],  # Horizontal details
                     coeffs_cropped[1][1][::2, ::2],  # Vertical details
                     coeffs_cropped[1][2][::2, ::2])  # Diagonal details

# To ensure compatibility, we'll pad the cropped coefficients
def pad_coeffs(coeffs, target_shape):
    new_coeffs = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            new_coeffs.append(tuple(np.pad(c, ((0, max(0, target_shape[0] - c.shape[0])),
                                               (0, max(0, target_shape[1] - c.shape[1]))), 
                                    mode='constant') for c in coeff))
        else:
            new_coeffs.append(np.pad(coeff, ((0, max(0, target_shape[0] - coeff.shape[0])),
                                             (0, max(0, target_shape[1] - coeff.shape[1]))), 
                                     mode='constant'))
    return new_coeffs

# Get target shape from the approximation coefficients
target_shape = coeffs[0].shape

# Pad the cropped coefficients
coeffs_cropped = pad_coeffs(coeffs_cropped, target_shape)

# Reconstruct the image from the cropped coefficients
reconstructed_image = pywt.waverec2(coeffs_cropped, 'haar')

# Plot original and reconstructed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Reconstructed Image (Cropped)')
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.savefig('explore/wave.png')