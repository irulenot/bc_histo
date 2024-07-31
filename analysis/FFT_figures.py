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
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = 'analysis/figures/FFT'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file, count):
    output_file = os.path.join(output_dir, str(count))
    if os.path.exists(f'{output_file}.npz'):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    image = grayscale_transform(image)

    fft_img = torch.fft.fft2(image)
    fft_img2 = torch.fft.fftshift(fft_img)
    crop_size = 1000
    h, w = fft_img.shape[-2], fft_img.shape[-1]
    start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    mask = torch.zeros(fft_img.shape)
    mask[:, start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
    fft_img_og = fft_img
    fft_img2_og = fft_img2
    fft_img = fft_img * mask
    fft_img2 = fft_img2 * mask
    rev_fft_img = torch.abs(torch.fft.ifft2(fft_img))
    rev_fft_img2 = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_img2)))

    height = 1000
    original_height = image.shape[1]
    original_width = image.shape[2]
    aspect_ratio = original_width / original_height
    width = int(height * aspect_ratio)
    new_shape = (height, width)

    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    resized_tensor = np.log(np.abs(rev_fft_img.squeeze().numpy()) + 1)
    axs[0, 0].imshow(resized_tensor, cmap='gray')
    axs[0, 0].set_title('reconstructed')
    axs[0, 0].axis('off')

    resized_tensor2 = np.log(np.abs(rev_fft_img2.squeeze().numpy()) + 1)
    axs[0, 1].imshow(resized_tensor2, cmap='gray')
    axs[0, 1].set_title('reconstructed shifted ')
    axs[0, 1].axis('off')

    fft_img = np.log(np.abs(fft_img.squeeze().numpy()) + 1)
    axs[1, 0].imshow(fft_img, cmap='gray')
    axs[1, 0].set_title('fft')
    axs[1, 0].axis('off')

    fft_img2 = np.log(np.abs(fft_img2.squeeze().numpy()) + 1)
    axs[1, 1].imshow(fft_img2, cmap='gray')
    axs[1, 1].set_title('fft shifted')
    axs[1, 1].axis('off')

    fft_img_og = np.log(np.abs(fft_img_og.squeeze().numpy()) + 1)
    axs[2, 0].imshow(fft_img_og, cmap='gray')
    axs[2, 0].set_title('fft og')
    axs[2, 0].axis('off')

    fft_img2_og = np.log(np.abs(fft_img2_og.squeeze() .numpy()) + 1)
    axs[2, 1].imshow(fft_img2_og, cmap='gray')
    axs[2, 1].set_title('fft shifted og')
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_file}.png')

    return output_file


tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

count = 0
for tiff_file in tqdm(tiff_files):
    if count >= 10:
        break
    result = process_tiff(tiff_file, count)
    count += 1
