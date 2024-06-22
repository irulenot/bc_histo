import os
import numpy as np
from tifffile import imread, imwrite
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from tifffile import imsave
from monai.data.wsi_reader import WSIReader
import matplotlib.pyplot as plt

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT_WSI/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    if os.path.exists(output_file):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    data_np = image.squeeze().numpy()

    if data_np.shape[1] > 20000 or data_np.shape[2] > 20000:
        return

    fft_channels = []
    for channel_data in data_np:
        fft_data = np.fft.fft2(channel_data)
        # fft_data_shifted = np.fft.fftshift(fft_data)
        fft_channels.append(fft_data.astype(np.complex64))
    fft_tensor = np.stack(fft_channels, axis=0)

    np.savez_compressed(output_file, array=fft_tensor, compression='gzip')
    return output_file

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)