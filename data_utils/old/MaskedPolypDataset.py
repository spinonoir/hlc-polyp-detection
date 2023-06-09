import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob as glob
import skimage.io as io
from  skimage.transform import resize
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


from src.config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE
from src.utils import collate_fn, get_transform

class MaskedPolypDataset(Dataset):
    def __init__(self, root, width, height, transforms=None):
        self.transforms = transforms
        self.root = root
        self.dir_images = os.path.join(self.root, "Original")
        self.dir_masks = os.path.join(self.root, "GroundTruth")
        self.width = width
        self.height = height


        self.all_images = list(sorted(glob.glob(os.path.join(self.dir_images, "*.tif"))))
        self.all_masks = list(sorted(glob.glob(os.path.join(self.dir_masks, "*.tif"))))
    
    def __getitem__(self, index):
        # Get image name and path
        img_path = self.all_images[index]
        mask_path = self.all_masks[index]
        img = io.imread(img_path)
        img = resize(img, (self.height, self.width))
        img = img.astype(np.float32) / 255.0

        mask_img = io.imread(mask_path)
        mask_img = resize(mask_img, (self.height, self.width))
        mask_img[mask_img < 1.] = 0.
        mask = mask_img.astype(np.uint8)
        
        
        # Get the boumding box coordinates for each mask
        num_objs = len(np.unique(mask))
        boxes = []
        for i in range(1, num_objs):
            pos = np.where(mask == i)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = (xmin, ymin, xmax, ymax)
            boxes.append(box)
        boxes = np.asarray(boxes, dtype=np.float32)
        # print(f'boxes shape: {boxes.shape}')
        # print(f'boxes shape: {boxes.shape}')

        labels = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        if self.transforms is not None:
            transformed = self.transforms(
                image=img, 
                bboxes=boxes,
                masks=[mask],
                labels=[labels]
            )

        target = {}
        target["boxes"] = transformed["bboxes"]
        target["labels"] = transformed["labels"]
        target["masks"] = transformed["masks"]
        target["image_id"] = image_id
        # target["area"] = area
        target["iscrowd"] = iscrowd


        # img = np.transpose(img, (2, 0, 1))
        img = transformed["image"]
        
        return img, target
    
    def __len__(self):
        return len(self.all_images)


    # def get_transform(self, train=True):
    #     bbox_p = A.BboxParams(
    #     format='pascal_voc',
    #     min_visibility=0.1,
    #     min_area=128, 
    #     label_fields=['labels'])
    
    #     if train:
    #         return A.Compose(
    #                 [
    #                     # TODO: Add more transformations and see what works best.
    #                     A.Flip(0.5),
    #                     A.RandomRotate90(0.5),
    #                     A.MotionBlur(p=0.5),
    #                     A.MedianBlur(blur_limit=3, p=0.1),
    #                     A.Blur(blur_limit=3, p=0.1),
    #                     ToTensorV2(p=1.0),
    #                 ], 
    #                 bbox_params=bbox_p,
    #             )

    #     else: 
    #         return A.Compose([
    #             ToTensorV2(p=1.0),
    #         ], bbox_params=bbox_p)