import os
import skimage.io
import random
import glob 
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_image_paths(dir_path):
    image_dir = os.path.join(dir_path, "Original")
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
    return image_paths

def get_mask_paths(dir_path):
    mask_dir = os.path.join(dir_path, "GroundTruth")
    mask_paths = glob.glob(os.path.join(mask_dir, "*.tif"))
    return mask_paths


def clamp_mask_colors(mask):
    mask[mask < 255] = 0
    mask[mask == 255] = 1
    return mask


# get bounding box coordinates for each mask
def get_mask_coordinates(mask):
    # get all object masks
    mask = clamp_mask_colors(mask)
    xmin = int(torch.min(mask[1]))
    xmax = int(torch.max(mask[1]))
    ymin = int(torch.min(mask[0]))
    ymax = int(torch.max(mask[0]))
    box = [xmin, ymin, xmax, ymax]
    return box

