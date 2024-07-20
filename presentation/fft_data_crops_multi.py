import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch
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
import torchvision.transforms as transforms

input_dir = '/data/breast-cancer/PANDA/train_images/'
tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for i, tiff_file in tqdm(enumerate(tiff_files)):
    
    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]
    transform = transforms.Grayscale(num_output_channels=1)
    image = transform(image).squeeze().numpy()

    # Plot and save the images side by side
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i3, i2 in enumerate([1000, 2000, 4000]):

        crop_size = i2
        h, w = image.shape[-2], image.shape[-1]
        start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2
        
        # Compute statistics
        image_stats = {
            'mean': image.mean(),
            'std': image.std(),
            'min': image.min(),
            'max': image.max()
        }

        # Perform FFT
        fft_image = fft2(image)
        fft_image = fft_image[start_h:start_h+crop_size, start_w:start_w+crop_size]
        fft_image_stats = {
        'mean': fft_image.mean(),
        'std': fft_image.std(),
        'min': fft_image.min(),
        'max': fft_image.max()
        }
        # Shift the FFT
        shifted_fft_image = fftshift(fft2(image))
        shifted_fft_image = shifted_fft_image[start_h:start_h+crop_size, start_w:start_w+crop_size]
        shifted_fft_image_stats = {
        'mean': shifted_fft_image.mean(),
        'std': shifted_fft_image.std(),
        'min': shifted_fft_image.min(),
        'max': shifted_fft_image.max()
        }

        # Helper function to add text
        def add_text(ax, stats):
            textstr = '\n'.join((
                f"Mean: {stats['mean']:.2f}",
                f"Std: {stats['std']:.2f}",
                f"Min: {stats['min']:.2f}",
                f"Max: {stats['max']:.2f}"
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        # Original image
        img1 = axes[i3][0].imshow(image, cmap='gray')
        axes[i3][0].set_title(f'Original Image, {tuple(image.shape)}')
        fig.colorbar(img1, ax=axes[0], orientation='vertical')
        add_text(axes[i3][0], image_stats)

        # FFT of the image
        img2 = axes[i3][1].imshow(np.log(np.abs(fft_image)), cmap='inferno')
        axes[i3][1].set_title(f'FFT_{i2}')
        fig.colorbar(img2, ax=axes[1], orientation='vertical')
        add_text(axes[i3][1], fft_image_stats)

        # Shifted FFT
        img3 = axes[i3][2].imshow(np.log(np.abs(shifted_fft_image)), cmap='inferno')
        axes[i3][2].set_title(f'Shifted FFT_{i2}')
        fig.colorbar(img3, ax=axes[2], orientation='vertical')
        add_text(axes[i3][2], shifted_fft_image_stats)
        
    # Save the plot
    plt.savefig(f'presentation/multi/fft_{i}_crop.png')
    print(f'presentation/multi/fft_{i}_crop.png')