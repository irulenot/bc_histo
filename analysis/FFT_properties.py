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
output_path = 'analysis/figures/FFT_properties.png'

def process_tiff(tiff_file, count, all_details):
    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"].to(float)
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    image = grayscale_transform(image)

    t1 = tuple(image.shape)
    t2 = torch.mean(image)
    t3 = torch.std(image)

    fft_img = torch.fft.fft2(image)
    fft_img2 = torch.fft.fftshift(fft_img)

    t11 = fft_img.shape
    t12 = torch.mean(fft_img)
    t13 = torch.std(fft_img)

    crop_size = 1000
    h, w = fft_img.shape[-2], fft_img.shape[-1]
    start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    fft_img_cropped = fft_img[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    fft_img2_cropped = fft_img2[:, start_h:start_h+crop_size, start_w:start_w+crop_size]

    t21 = fft_img2.shape
    t22 = torch.mean(fft_img_cropped)
    t23 = torch.std(fft_img_cropped)

    t31 = fft_img2.shape
    t32 = torch.mean(fft_img2_cropped)
    t33 = torch.std(fft_img2_cropped)

    height = 1000
    original_height = image.shape[1]
    original_width = image.shape[2]
    aspect_ratio = original_width / original_height
    width = int(height * aspect_ratio)
    new_shape = (height, width)

    image = F.interpolate(image.unsqueeze(0), size=new_shape, mode='bilinear', align_corners=False).squeeze()
    
    axs[count].text(0.5, 0.02, (
        f"Image\n"
        f"Shape: {tuple(t1)}\n"
        f"Mean: {t2.item():.2e}\n"
        f"Std: {t3.item():.2e}\n\n"
        f"FFT Image\n"
        f"Mean: {t12.item():.2e}\n"
        f"Std: {t13.item():.2e}\n\n"
        f"Cropped FFT Image\n"
        f"Mean: {t22:.2e}\n"
        f"Std: {t23.item():.2e}\n"
        f"Cropped Shifted FFT Image\n"
        f"Mean: {t32.item():.2e}\n"
        f"Std: {t33.item():.2e}"
    ), ha='center', va='center', verticalalignment='bottom', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs[count].imshow(image, cmap='gray')
    axs[count].set_title(str(count))
    axs[count].axis('off')

    details = {
        'image': {'shape': str(t1), 'mean': t2.item(), 'std': t3.item()},
        'fft': {'mean': t12.real.item(), 'std': t13.real.item()},
        'fft_cropped': {'mean': t22.real.item(), 'std': t23.real.item()},
        'fft_cropped_shifted': {'mean': t32.real.item(), 'std': t33.real.item()},
    }
    all_details[count] = details
    return None

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

count = 0
all_details = {}
fig, axs = plt.subplots(1, 10, figsize=(60, 12))
for tiff_file in tqdm(tiff_files):
    if count >= 10:
        break
    result = process_tiff(tiff_file, count, all_details)
    count += 1

plt.tight_layout()
plt.savefig(f'{output_path}.png')

import json
file_path = 'analysis/figures/FFT_properties.json'
with open(file_path, 'w') as json_file:
    json.dump(all_details, json_file, indent=4)  # indent=4 for pretty printing
