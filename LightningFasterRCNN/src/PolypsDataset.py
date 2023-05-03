from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset

from torchvision import io, utils
# from torchvision import datapoints
from torchvision.transforms import functional as F

import cv2

import os
import pandas as pd
import glob as glob

import numpy as np
# from torchvision.transforms import v2 as T
# from torchvision.transforms.v2 import functional as F

from src.config import CLASSES, RESIZE_TO 


class PolypsDataset(Dataset):
    def __init__(self, root_dir, stage='train', resize_to=800, transforms=None):
        # make sure state is valid
        assert stage in ['train', 'validation', 'test'], f'Stage {stage} not recognized'

        # make sure root_dir exists
        assert os.path.exists(root_dir), f'Root directory {root_dir} does not exist'
        self.root_dir = root_dir

        # set transforms
        self.transforms = transforms

        # set resize_to
        self.resize_to = resize_to

        # Get csv file path, make sure it exists, and read it into a pandas dataframe
        csv_path = os.path.join(self.root_dir, 'all_data.csv')
        assert os.path.exists(csv_path), f'CSV file {csv_path} does not exist'
        self.df_annotations = pd.read_csv(csv_path)

        # Get the image paths, make sure the list is not empty
        self.stage_images = list(sorted(glob.glob(os.path.join(self.root_dir, "*", stage, "images", "*.[jp][pn]g"))))
        assert len(self.stage_images) > 0, f'No images found for stage {stage}'
    
    def __len__(self):
        return len(self.stage_images)
    
    def __getitem__(self, idx):
        image_path = self.stage_images[idx]

        # assert image_path exists
        assert os.path.exists(image_path), f'Image path {image_path} does not exist'        

        # resize image using torchvision
        # image = io.read_image(image_path)
        # image = F.resize(image, [self.resize_to['height'], self.resize_to['width']])
        # image = np.asarray(image, dtype=np.float32) / 255.0

        # load and resize image using cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height = image.shape[0]
        img_width = image.shape[1]
        image = cv2.resize(image, (self.resize_to, self.resize_to))
        image = np.asarray(image, dtype=np.float32) / 255.0

        # get annotations for this image
        annotation = self.df_annotations.loc[self.df_annotations['path'] == image_path.strip(self.root_dir+'/')]

        # get class labels
        labels = annotation['label'].values.tolist()

        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # get bounding boxes, and resize them
        xmin = annotation['xmin'].values.tolist()
        ymin = annotation['ymin'].values.tolist()
        xmax = annotation['xmax'].values.tolist()
        ymax = annotation['ymax'].values.tolist()
        boxes = []
        for i in range(len(xmin)):
            box = [
                (xmin[i] / img_width) * self.resize_to['width'],
                (ymin[i] / img_height) * self.resize_to['height'],
                (xmax[i] / img_width) * self.resize_to['width'],
                (ymax[i] / img_height) * self.resize_to['height']
            ]
            if box[0] >= box[2] or box[1] >= box[3]:
                box = [2, 2, 7, 7]
            boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # make sure the box dimensions are correct
    
        
        # Apply transforms, if they exist
        if self.transforms is not None:
            sample = self.transforms(
                image=image, 
                bboxes=boxes, 
                labels=labels
            )
        


        # Make sure a box exists
        if len(sample['bboxes']) == 0:
            box = [2, 2, 7, 7]
            boxes = torch.as_tensor(box, dtype=torch.float32)
            labels = torch.as_tensor([0], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        
        image = torch.as_tensor(sample['image'], dtype=torch.float32)

        target = {}
        target['boxes'] = boxes.unsqueeze(0)
        target['labels'] = labels

        return image, target