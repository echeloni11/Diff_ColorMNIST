import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from matplotlib.colors import hsv_to_rgb

import torch.utils.data as Data
from torchvision import datasets, transforms

def add_hue_confounded(images, labels, sigma=0.05, p_unif=0):
    device = images.device
    colored_images = []
    hues = []
    for img, label in zip(images, labels):
        # Compute the mean hue for the given label
        is_unif = np.random.rand() < p_unif
        if is_unif:
            hue = np.random.rand()
        else:
            mean_hue = label.detach().cpu() / 10 + 0.05
            # Draw a hue from the normal distribution
            hue = np.random.normal(loc=mean_hue, scale=sigma)
        hue = hue % 1  # Ensure the hue is between 0 and 1

        # Convert the grayscale image to HSV (Hue, Saturation, Value)
        # The original image is just the Value channel
        hsv_image = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
        hsv_image[..., 0] = hue
        hsv_image[..., 1] = 1.0  # Full saturation
        hsv_image[..., 2] = img.detach().cpu().squeeze()  # Value channel is the grayscale image

        # Convert HSV to RGB
        rgb_image = hsv_to_rgb(hsv_image).transpose(2, 0, 1)
        
        colored_images.append(rgb_image)
        hues.append(hue)
    colored_images = np.array(colored_images)
    return torch.tensor(colored_images, dtype=torch.float32, device=device), torch.tensor(hues, dtype=torch.float32, device=device)


class classifiedMNIST(datasets.MNIST):
    pass