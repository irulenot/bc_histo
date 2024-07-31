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

shapes = []
def process_tiff(tiff_file):
    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=0, image_only=True)
    image_dict = loader({"image": os.path.join(input_dir, tiff_file)})
    image = image_dict["image"]

    shapes.append(np.array(image.shape[1:]))

tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for tiff_file in tqdm(tiff_files):
    result = process_tiff(tiff_file)

shapes = np.array(shapes)
statistics = {
    'count': shapes.shape[0],
    'min': str(shapes.min()),
    'max': str(shapes.max()),
    'mean': str(shapes.mean()),
    'std': str(shapes.std()),
    'median': str(np.median(shapes)),
}

# Save statistics to a JSON file
with open('explore/shape_statistics.json', 'w') as f:
    json.dump(statistics, f, indent=4)

print("Statistics saved to shape_statistics.json")