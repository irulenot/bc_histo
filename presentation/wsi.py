import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch
from PIL import Image


image = np.load('presentation/WSI_image.npz')['arr_0']
image = image.transpose(1, 2, 0)
image_pil = Image.fromarray(image)
image_pil.save('saved_image.png')