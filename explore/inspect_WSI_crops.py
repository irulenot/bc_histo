import os
import numpy as np
from tifffile import imread, imwrite
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from skimage.io import imsave
from monai.data.wsi_reader import WSIReader
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
import torch
import torch.fft
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import gzip
import shutil
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = 'outputs/inspect/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    if os.path.exists(f'{output_file}.npz'):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]

    C, H, W = image.shape
    if W > H:
        image = image.permute(0, 2, 1)
    C, H, W = image.shape
    if W < H:
        short_image = image.permute(0, 2, 1)

    transform = transforms.ToPILImage()
    # img = transform(image)
    # img.save('outputs/inspect/img.png')
    # img = transform(short_image)
    # img.save('outputs/inspect/short_img.png')

    fft_img = torch.fft.fft2(image)
    fft_img_shifted = torch.fft.fftshift(fft_img)
    # img = transform(fft_img.real)
    # img.save('outputs/inspect/fft_img.png')
    # img = transform(fft_img_shifted.real)
    # img.save('outputs/inspect/fft_img_shifted.png')

    # inv_fft_img = torch.fft.ifft2(fft_img)
    # inv_fft_img_shifted = torch.fft.ifft2(torch.fft.ifftshift(fft_img_shifted))
    # img = transform(inv_fft_img.real)
    # img.save('outputs/inspect/inv_inv_fft_img.png')
    # img = transform(inv_fft_img_shifted.real)
    # img.save('outputs/inspect/inv_inv_fft_img_shifted.png')    

    crop_sizes = [1000]
    h, w = fft_img.shape[-2], fft_img.shape[-1]

    for crop_size in crop_sizes:
        image = fft_img
        h = image.shape[1]
        start_h = h // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, :] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = fft_img_shifted
        h = image.shape[1]
        start_h = h // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, :] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift((image)))
        img = transform(image.real)
        img.save(f'outputs/inspect/cropped_shifted_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img
        h = image.shape[1]
        start_h = h // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, :] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/short_cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img_shifted
        h = image.shape[1]
        start_h = h // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, :] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift(image))
        img = transform(image.real)
        img.save(f'outputs/inspect/short_cropped_shifted_image_{crop_size}.png')

# ########### WIDTH

    for crop_size in crop_sizes:
        image = fft_img
        h = image.shape[2]
        start_h = h // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, :, start_h:start_h+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/width_cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = fft_img_shifted
        h = image.shape[2]
        mask = torch.zeros(image.shape)
        mask[:, :, start_h:start_h+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift(image))
        img = transform(image.real)
        img.save(f'outputs/inspect/width_cropped_shifted_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img
        h = image.shape[2]
        mask = torch.zeros(image.shape)
        mask[:, :, start_h:start_h+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/width_short_cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img_shifted
        h = image.shape[2]
        mask = torch.zeros(image.shape)
        mask[:, :, start_h:start_h+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift(image))
        img = transform(image.real)
        # img.save(f'outputs/inspect/width_short_cropped_shifted_image_{crop_size}.png')

################## Square
    for crop_size in crop_sizes:
        image = short_fft_img
        h, w = image.shape[1], image.shape[2]
        start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/square_short_cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img_shifted
        h, w = image.shape[1], image.shape[2]
        start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        image = image * mask        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift(image))
        img = transform(image.real)
        img.save(f'outputs/inspect/square_short_cropped_shifted_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img

        c, h, w = image.size()
        half_h, half_w = h // 2, w // 2
        # Create an empty image with the same size
        inverted_image = torch.empty_like(image)
        for channel in range(c):
            # Top-left to center
            inverted_image[channel, half_h:, half_w:] = image[channel, :half_h, :half_w]
            # Top-right to bottom-left
            inverted_image[channel, half_h:, :half_w] = image[channel, :half_h, half_w:]
            # Bottom-left to top-right
            inverted_image[channel, :half_h, half_w:] = image[channel, half_h:, :half_w]
            # Bottom-right to top-left
            inverted_image[channel, :half_h, :half_w] = image[channel, half_h:, half_w:]
        image = inverted_image

        h, w = image.shape[1], image.shape[2]
        start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        image = image * mask
        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(image)
        img = transform(image.real)
        img.save(f'outputs/inspect/inv_square_short_cropped_image_{crop_size}.png')

    for crop_size in crop_sizes:
        image = short_fft_img_shifted

        c, h, w = image.size()
        half_h, half_w = h // 2, w // 2
        # Create an empty image with the same size
        inverted_image = torch.empty_like(image)
        for channel in range(c):
            # Top-left to center
            inverted_image[channel, half_h:, half_w:] = image[channel, :half_h, :half_w]
            # Top-right to bottom-left
            inverted_image[channel, half_h:, :half_w] = image[channel, :half_h, half_w:]
            # Bottom-left to top-right
            inverted_image[channel, :half_h, half_w:] = image[channel, half_h:, :half_w]
            # Bottom-right to top-left
            inverted_image[channel, :half_h, :half_w] = image[channel, half_h:, half_w:]
        image = inverted_image

        h, w = image.shape[1], image.shape[2]
        start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        mask = torch.zeros(image.shape)
        mask[:, start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        image = image * mask        # Perform inverse FFT and convert to numpy array
        image = torch.fft.ifft2(torch.fft.ifftshift(image))
        img = transform(image.real)
        img.save(f'outputs/inspect/inv_square_short_cropped_shifted_image_{crop_size}.png')

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

result = process_tiff(tiff_files[9])


# import torch
# import torchvision.transforms.functional as TF
# from PIL import Image
# # Assuming cropped_fft_img is your cropped Fourier transformed image tensor
# # cropped_fft_img should be complex valued
# # Apply Inverse Fourier Transform
# reconstructed_img_complex = torch.fft.ifft2(cropped_fft_img, dim=(-2, -1))
# # Take the real and imaginary parts
# reconstructed_real = torch.real(reconstructed_img_complex).detach().cpu()
# reconstructed_imag = torch.imag(reconstructed_img_complex).detach().cpu()
# # Combine real and imaginary parts to reconstruct the image
# reconstructed_img = reconstructed_real + 1j * reconstructed_imag
# # Take absolute value to get magnitude
# magnitude_spectrum = torch.abs(reconstructed_img)
# # Normalize to [0, 1] range (assuming original was normalized to [0, 1])
# magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
# # Convert tensor to PIL Image
# reconstructed_img = TF.to_pil_image(magnitude_spectrum.squeeze(0))
# # Save the reconstructed image
# save_path = 'reconstructed_image.jpg'
# reconstructed_img.save(save_path)
# print(f"Reconstructed image saved at: {save_path}")
