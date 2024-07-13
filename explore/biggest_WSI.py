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
import json
import torch.nn.functional as F

input_dir = '/data/breast-cancer/PANDA/train_images/'

largest_size, largest_path = 0, 0
def process_tiff(tiff_file, largest_size, largest_path):
    size = os.path.getsize(input_dir + tiff_file)
    if size > largest_size:
        largest_size = size
        largest_path = tiff_file
    return largest_size, largest_path

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    largest_size, largest_path = process_tiff(tiff_file, largest_size, largest_path)

loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
image_dict = loader({"image": os.path.join(input_dir, largest_path)})
image = image_dict["image"]
print(image.shape)