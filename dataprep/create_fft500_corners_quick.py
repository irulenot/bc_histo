import os
import numpy as np
from tifffile import imread, imsave
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from tifffile import imsave
from monai.data.wsi_reader import WSIReader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import json

input_dir = '/data/breast-cancer/PANDA/train_images_FFT_WSI/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT500_corners_mid/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    input_file = os.path.join(input_dir, tiff_file)
    output_file = os.path.join(output_dir, tiff_file)
    if os.path.exists(output_file):
        return 0

    try:
        image = np.load(input_file)['array']
    except Exception as e:
        return 1
    
    fft_channels = []
    size = 500
    for channel_data in image:
        mid_start1 = (channel_data.shape[0] // 2) - size//2
        mid_end1 = (channel_data.shape[0] // 2) + size//2
        mid_start2 = (channel_data.shape[1] // 2) - size//2
        mid_end2 = (channel_data.shape[1] // 2) + size//2
        middle = channel_data[mid_start1:mid_end1, mid_start2:mid_end2]
        top_left = channel_data[:size, :size]
        top_right = channel_data[:size, -size:]
        bottom_left = channel_data[-size:, :size]
        bottom_right = channel_data[-size:, -size:]
        fft_tensor_channel = np.stack([middle, top_left, top_right, bottom_left, bottom_right], axis=0)
        fft_channels.append(fft_tensor_channel)
    fft_tensor = np.stack(fft_channels, axis=0)

    np.savez_compressed(output_file, array=fft_tensor, compression='gzip')
    return output_file

tiff_files = [f for f in os.listdir(input_dir)]

json_file_path = 'datalists/datalist_panda_fft_quick.json'
with open(json_file_path, 'r') as file:
    datalist_panda_fft_quick = json.load(file)
json_filenames = set()
for entry in datalist_panda_fft_quick['training']:
    json_filenames.add(entry['image'][:-5] + '.npz')
for entry in datalist_panda_fft_quick['validation']:
    json_filenames.add(entry['image'][:-5] + '.npz')
filtered_tiff_files = [filename for filename in tiff_files if filename in json_filenames]

max_threads = 3
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    results = list(tqdm(executor.map(process_tiff, filtered_tiff_files), total=len(filtered_tiff_files)))

# # Verifying images
# # Original
# img_transposed = np.transpose(image, (1, 2, 0))
# plt.imsave('outputs/original_image.png', img_transposed , cmap='gray')
# # FFT
# magnitude_spectrum = np.log(np.abs(fft_tensor) + 1)
# img_transposed = np.transpose(magnitude_spectrum, (1, 2, 0))
# img_normalized = (img_transposed.real - img_transposed.real.min()) / (img_transposed.real.max() - img_transposed.real.min())
# plt.imsave('outputs/fft_image.png', img_normalized, cmap='gray')
# # Reconstruction
# reconstructed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_tensor)))
# img_transposed = np.transpose(reconstructed_image, (1, 2, 0))
# img_normalized = (img_transposed.real - img_transposed.real.min()) / (img_transposed.real.max() - img_transposed.real.min())
# plt.imsave('outputs/reconstructed_image.png', img_normalized, cmap='gray')
# #