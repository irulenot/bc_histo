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

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = '/data/breast-cancer/PANDA/train_images_FFT500_WSI_grayscaled/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    if os.path.exists(output_file):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    image = image.squeeze().numpy()

    if image.shape[1] > 20000 or image.shape[2] > 20000:
        return

    target_size = (20000, 20000) # pad don't resize, w 255?
    pad_height = target_size[0] - image.shape[1]
    pad_width = target_size[1] - image.shape[2]
    padded_image = np.pad(image.transpose(1, 2, 0), ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=255)
    gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    DFT = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(DFT)

    row, col = gray_image.shape
    center_row, center_col = row // 2, col // 2
    crop_size = 2000 
    start_row = center_row - crop_size // 2
    end_row = center_row + crop_size // 2
    start_col = center_col - crop_size // 2
    end_col = center_col + crop_size // 2
    cropped_shift = shift[start_row:end_row, start_col:end_col]
    
    np.savez_compressed(output_file, array=cropped_shift, compression='gzip')
    return output_file

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)

# # REVERT USING NUMPY
# # Perform inverse DFT
# cropped_ifft_shift = np.fft.ifftshift(cropped_shift)
# imageThen = cv2.idft(cropped_ifft_shift)
# # Calculate the magnitude of the inverse DFT
# imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
# # Visualize the original image and the magnitude spectrum
# plt.figure(figsize=(10,10))
# plt.subplot(121), plt.imshow(gray_image, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(imageThen, cmap='gray')
# plt.title('Magnitude Spectrum (Cropped)'), plt.xticks([]), plt.yticks([])
# # Save the plot as a PNG file
# plt.savefig('magnitude_spectrum_cropped.png')

# REVERT USING PYTORCH
# cropped_shift_tensor = torch.tensor(cropped_shift)
# shift_back = torch.fft.ifftshift(cropped_shift_tensor, dim=(0, 1))
# idft = torch.fft.ifft2(shift_back, dim=(0, 1))
# idft_magnitude = torch.abs(idft)
# # Normalize the magnitude to the range [0, 255] for image saving
# idft_magnitude = idft_magnitude - idft_magnitude.min()
# idft_magnitude = idft_magnitude / idft_magnitude.max() * 255.0
# idft_magnitude = idft_magnitude.to(torch.uint8)
# # Convert to a PIL image
# idft_image = to_pil_image(idft_magnitude[:,:,0])  # Squeeze to remove extra dimensions
# # Save the image as PNG
# output_path = "output.png"
# idft_image.save(output_path)

