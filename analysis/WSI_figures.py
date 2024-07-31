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
from torchvision import transforms
from PIL import Image

input_dir = '/data/breast-cancer/PANDA/train_images/'
output_dir = 'analysis/figures/WSI'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file, count):
    output_file = os.path.join(output_dir, str(count))
    if os.path.exists(f'{output_file}.npz'):
        return 0

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]

    height = 1000
    original_height = image.shape[1]
    original_width = image.shape[2]
    aspect_ratio = original_width / original_height
    width = int(height * aspect_ratio)
    resize_transform = transforms.Resize((height, width))
    resized_tensor = resize_transform(image)

    image = Image.fromarray(resized_tensor.permute(1, 2, 0).numpy())
    image.save(f'{output_file}.png')
    return output_file


tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

count = 0
for tiff_file in tqdm(tiff_files):
    if count >= 10:
        break
    result = process_tiff(tiff_file, count)
    count += 1
