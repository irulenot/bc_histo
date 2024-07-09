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
import torch
import torchvision.transforms.functional as TF
from PIL import Image

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = 'outputs/inspect4/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    output_file = os.path.join(output_dir, tiff_file)[:-5]
    # if os.path.exists(f'{output_file}.npz'):
    #     return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    image = image.squeeze().numpy()

    if image.shape[1] > 20000 or image.shape[2] > 20000:
        print('FAIL')
        return

    target_size = (20000, 20000) # pad don't resize, w 255?
    pad_height = target_size[0] - image.shape[1]
    pad_width = target_size[1] - image.shape[2]
    padded_image = np.pad(image.transpose(1, 2, 0), ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
    
    save_path = os.path.join(output_dir, f'padded_image.jpg')
    cv2.imwrite(save_path, padded_image)
    
    gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)

    save_path = os.path.join(output_dir, f'gray_image.jpg')
    cv2.imwrite(save_path, gray_image)

    resized_gray = cv2.resize(gray_image, (5000, 5000), interpolation=cv2.INTER_CUBIC)
    save_path = os.path.join(output_dir, f'resized_gray_image.jpg')
    cv2.imwrite(save_path, resized_gray)

    gray_image_tensor = torch.tensor(gray_image, dtype=torch.float32)
    fft_img = torch.fft.fft2(gray_image_tensor, dim=(-2, -1))
    fft_img = torch.fft.fftshift(fft_img, dim=(-2, -1))


    crop_sizes = [1000, 2000, 4000, 6000, 9000 , 10000, 11000]
    for crop_size in crop_sizes:
        h, w = fft_img.shape[-2], fft_img.shape[-1]
        start_h = h // 2 - crop_size // 2
        start_w = w // 2 - crop_size // 2
        
        # Create a mask to zero out the middle area
        mask = torch.zeros(fft_img.shape)
        mask[:crop_size//4] = 1
        mask[-crop_size//4:] = 1
        # mask[:, :crop_size//4] = 1
        # mask[:, -crop_size//4:] = 1
        # mask[start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        
        cropped_fft_img = fft_img * mask

        # Perform inverse FFT and convert to numpy array
        cropped_fft_img = torch.fft.ifft2(torch.fft.ifftshift(cropped_fft_img, dim=(-2, -1)), dim=(-2, -1))
        cropped_fft_img = cropped_fft_img.squeeze().cpu().numpy()
        
        # Convert to PIL image and save
        pil_image = Image.fromarray(cropped_fft_img.astype('uint8'))
        save_path = os.path.join(output_dir, f'top_left_reconstructed_image_{crop_size}.jpg')
        pil_image.save(save_path)

    crop_sizes = [1000, 2000, 4000, 6000, 9000 , 10000, 11000]
    for crop_size in crop_sizes:
        h, w = fft_img.shape[-2], fft_img.shape[-1]
        start_h = h // 2 - crop_size // 2
        start_w = w // 2 - crop_size // 2
        
        # Create a mask to zero out the middle area
        mask = torch.zeros(fft_img.shape)
        mask[:crop_size//4] = 1
        # mask[-crop_size//4:] = 1
        # mask[:, :crop_size//4] = 1
        mask[:, -crop_size//4:] = 1
        # mask[start_h:start_h+crop_size, start_w:start_w+crop_size] = 1
        
        cropped_fft_img = fft_img * mask

        # Perform inverse FFT and convert to numpy array
        cropped_fft_img = torch.fft.ifft2(torch.fft.ifftshift(cropped_fft_img, dim=(-2, -1)), dim=(-2, -1))
        cropped_fft_img = cropped_fft_img.squeeze().cpu().numpy()
        
        # Convert to PIL image and save
        pil_image = Image.fromarray(cropped_fft_img.astype('uint8'))
        save_path = os.path.join(output_dir, f'top_halves_reconstructed_image_{crop_size}.jpg')
        pil_image.save(save_path)


tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

result = process_tiff(tiff_files[9])


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
