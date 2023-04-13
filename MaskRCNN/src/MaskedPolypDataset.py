import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob as glob
import skimage.io as io
from  skimage.transform import resize

from src.config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE
from src.utils import collate_fn, get_transform

class MaskedPolypDataset(Dataset):
    def __init__(self, root, width, height, classes, transforms=None):
        self.transforms = transforms
        self.root = root
        self.dir_images = os.path.join(self.root, "Original")
        self.dir_masks = os.path.join(self.root, "GroundTruth")
        self.width = width
        self.height = height
        self.classes = classes

        self.all_images = list(sorted(glob.glob(os.path.join(self.dir_images, "*.tif"))))
        self.all_masks = list(sorted(glob.glob(os.path.join(self.dir_masks, "*.tif"))))
    
    def __getitem__(self, index):
        # Get image name and path
        img_path = self.all_images[index]
        mask_path = self.all_masks[index]
        img = io.imread(img_path)
        img = resize(img, (self.height, self.width))

        mask_img = io.imread(mask_path)
        mask_img = resize(mask_img, (self.height, self.width))
        mask = clamp_mask_colors(mask_img)

        bbox = get_mask_coordinates(mask)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)

        labels = torch.ones((1,), dtype=torch.int64)
        boxes = torch.as_tensor([bbox], dtype=torch.float32)
        masks = torch.as_tensor([mask], dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
    
        return img, target
    
    def __len__(self):
        return len(self.all_images)


def get_mask_coordinates(mask):
    # get all object masks
    xmin = int(torch.min(mask[1]))
    xmax = int(torch.max(mask[1]))
    ymin = int(torch.min(mask[0]))
    ymax = int(torch.max(mask[0]))
    box = [xmin, ymin, xmax, ymax]
    return box

def clamp_mask_colors(mask):
    mask[mask < 255] = 0
    mask[mask == 255] = 1
    return mask