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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT1000n_WSI_both_centered/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    if os.path.exists(f'{output_file}.npz'):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]

    if image.shape[1] > 50000 or image.shape[2] > 50000:
        return 0

    fft_img = torch.fft.fftn(image)
    fft_img2 = torch.fft.fftshift(fft_img)

    crop_size = 1000
    h, w = fft_img.shape[-2], fft_img.shape[-1]
    start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    
    fft_img = fft_img[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    fft_img2 = fft_img2[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    if fft_img.shape[0] != 3 or fft_img.shape[1] != crop_size or fft_img.shape[2] != crop_size:
        return

    fft_img = torch.stack([fft_img, fft_img2])
    np.savez_compressed(f'{output_file}', fft_img.numpy())
    return output_file


tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]
file_names = os.listdir(output_dir)
new_file_names = []
for filename in file_names:
    if filename.endswith('.npz'):
        new_filename = filename.split('.')[0] + '.tiff'  # Remove '.npz' and add '.tiff'
    else:
        new_filename = filename  # Keep original filename if it doesn't end with .npz
    new_file_names.append(new_filename)
highest_index = 0
for filename in new_file_names:
    if filename in tiff_files:
        index = tiff_files.index(filename)
        if index > highest_index:
            highest_index = index
tiff_files = tiff_files[highest_index:]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)


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
