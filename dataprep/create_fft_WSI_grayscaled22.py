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

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT_WSI_grayscaled2/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    if os.path.exists(f'{output_file}.npz'):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    image = image.squeeze().numpy()

    if image.shape[1] > 20000 or image.shape[2] > 20000:
        return

    # target_size = (20000, 20000) # pad don't resize, w 255?
    # pad_height = target_size[0] - image.shape[1]
    # pad_width = target_size[1] - image.shape[2]
    padded_image = np.pad(image.transpose(1, 2, 0), ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
    gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    gray_image_tensor = torch.tensor(gray_image, dtype=torch.float32)
    fft_img = torch.fft.fft2(gray_image_tensor, dim=(-2, -1))

    # mask = torch.zeros(fft_img.shape)
    # crop_size = 2000
    # quarter = crop_size//4
    # mask[:quarter, :mask.size(1)-quarter] = 255
    # mask[-quarter:, quarter:] = 255
    # mask[:mask.size(0)-quarter, -quarter:] = 255
    # mask[quarter:, :quarter] = 255

    crop_size = 2000
    quarter = crop_size//4
    Top = fft_img[:quarter, :fft_img.size(1)-quarter]
    Bottom = fft_img[-quarter:, quarter:]
    Right = fft_img[:fft_img.size(0)-quarter, -quarter:]
    Left = fft_img[quarter:, :quarter]
    # Apply padding only along dimension 1
    Top = F.pad(Top, (0, quarter), "constant", 0)
    Bottom = F.pad(Bottom, (quarter, 0), "constant", 0)
    Right = F.pad(Right, (0, 0, quarter, 0), "constant", 0)
    Left = F.pad(Left, (0, 0, 0, quarter), "constant", 0)
    combined = torch.cat([Top, Bottom, Right, Left], dim=0)
    
    np.savez_compressed(f'{output_file}.npz', combined.numpy())
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
