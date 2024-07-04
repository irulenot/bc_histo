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
output_dir = '/data/breast-cancer/PANDA/train_images_FFT750_shifted/'
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
    radius = 750
    for channel_data in image:
        rows, cols = channel_data.shape
        crow, ccol = rows // 2, cols // 2
        fft_data_shifted = np.fft.fftshift(channel_data)
        cropped_fft_data = fft_data_shifted[crow - radius:crow + radius, ccol - radius:ccol + radius]
        fft_channels.append(cropped_fft_data)
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