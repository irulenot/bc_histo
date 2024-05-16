import os
import numpy as np
from tifffile import imread, imsave
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from tifffile import imsave
from monai.data.wsi_reader import WSIReader
import matplotlib.pyplot as plt

input_dir = '/data/breast-cancer/PANDA/train_images_FFT/'
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)

    from PIL import Image
    image = imread(input_dir + tiff_file)

    image = np.squeeze(image)
    reconstructed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(image)))
    img_transposed = np.transpose(reconstructed_image, (1, 2, 0))
    img_normalized = (img_transposed.real - img_transposed.real.min()) / (img_transposed.real.max() - img_transposed.real.min())
    plt.imsave('outputs/temp.png', img_normalized, cmap='gray')

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)