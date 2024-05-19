import os
import numpy as np
from tifffile import imread, imsave
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from tifffile import imsave
from monai.data.wsi_reader import WSIReader
import matplotlib.pyplot as plt
import cv2

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT_gray/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)
    if os.path.exists(output_file):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    
    if image.shape[-1] < 1024 or image.shape[-2] < 1024:
        return None
    if image.shape[-1] > 25000 or image.shape[-2] > 25000:
        return None
    
    image = np.transpose(image.numpy(), (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    radius = 1000
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fft_data = np.fft.fft2(image)
    fft_data_shifted = np.fft.fftshift(fft_data)
    fft_tensor = fft_data_shifted[crow - radius:crow + radius, ccol - radius:ccol + radius]

    fft_tensor = np.expand_dims(fft_tensor, axis=0)
    fft_tensor = np.pad(fft_tensor, ((0, 0), (24, 24), (24, 24)), mode='constant', constant_values=0)

    imsave(output_file, fft_tensor)
    return output_file

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)