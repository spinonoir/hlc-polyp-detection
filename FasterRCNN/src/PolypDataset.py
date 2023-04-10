from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import numpy as np
import os
import glob as glob
import xmltodict
import pandas as pd

from src.config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE
from src.utils import collate_fn, get_transform
from torch.utils.data import Dataset, DataLoader

class PolypDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.dir_images = os.path.join(self.dir_path, "images")
        self.width = width
        self.height = height
        self.classes = classes

        # Read the labels in as a pandas dataframe
        label_path = glob.glob(os.path.join(self.dir_path, "*labels.csv"))
        self.csv_labels = pd.read_csv(label_path[0])

        # Define the directory where the images are stored
        # image_dir = os.path.join(self.dir_path, "images")

        # Get the image paths
        # self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        # self.all_images = [image_path.split("/")[-1] for image_path in self.image_paths]
        # self.all_images = sorted(self.all_images)


    def __getitem__(self, index):
        # Get image name and path
        image_name = self.csv_labels.iloc[index]['filename']
        image_path = os.path.join(self.dir_images, image_name)

        # Load image
        image = cv2.imread(image_path)
        # For some reason many examples in the documentation indicate that we need to convert
        # the image from BGR to RGB. This was not my experience, but I left the artifact here
        # as a reminder for when we add more images to the dataset.
        # TODO: When adding new data, verify it loads as RGB and add a TAG to the XML 
        # file to indicate that the image is RGB vs BGR if needed.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = image_resized.astype(np.float32) / 255.0

        # Get image dimensions
        image_height, image_width, _ = image.shape

        # Get label
        annotation_filename = image_name[:-4] + ".xml"
        annotation_file_path = os.path.join(self.dir_path, annotation_filename)

        # Some nice containers to hold the bounding boxes and labels <3
        # boxes = []
        # labels = []

       

        # with open(annotation_file_path, "r") as f:
        #     # Read the label data from the XML file
        #     doc = xmltodict.parse(f.read())

        # # When we have multiple objects, the parser will return a list of dictionaries
        # # with the corresponding labels and bboxes. If it's a single dictionary, we will
        # # add it to a list to make it easier to iterate over.
        # if not isinstance(doc["annotation"]["object"], list):
        #     objs = [doc["annotation"]["object"]]
        # else:
        #     objs = doc["annotation"]["object"]

        # for obj in objs:
        #     # In order to avoiud issues with Albumentations trying to draw bounding boxes
        #     # on images with no polyps, we will create a dummy bounding box for images
        #     # that has smaller area than the min_area param passed into the bbox_params.
        #     labels.append(self.classes.index(obj["name"]))
        #     if obj['name'] == CLASSES[0]:
        #         xmin, ymin, xmax, ymax = int(1), int(1), int(3), int(3)
        #     else:
        #         xmin = int(obj["bndbox"]["xmin"])
        #         ymin = int(obj["bndbox"]["ymin"])
        #         xmax = int(obj["bndbox"]["xmax"])
        #         ymax = int(obj["bndbox"]["ymax"])

        #     xmin_resized = (xmin/image_width) * self.width
        #     ymin_resized = (ymin/image_height) * self.height
        #     xmax_resized = (xmax/image_width) * self.width
        #     ymax_resized = (ymax/image_height) * self.height
        #     boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])

        # get the label info from the pandas dataframe
        label_info = self.csv_labels.loc[self.csv_labels['filename'] == image_name]
        labels = label_info['class'].values.tolist()
        labels = [self.classes.index(label) for label in labels]

        xmin = (label_info['xmin']/image_width) * self.width
        ymin = (label_info['ymin']/image_height) * self.height
        xmax = (label_info['xmax']/image_width) * self.width
        ymax = (label_info['ymax']/image_height) * self.height

        boxes = np.stack((xmin, ymin, xmax, ymax), axis=1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Convert boxes to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Calculate the area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Crowd instances -- will be useful for mulibox training
        iscrowd = torch.zeros((boxes.shape[0]), dtype=torch.int64)

        # Convert labels to torch tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create a target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        # Apply transformations using Albumentations
        if self.transforms:
            sample = self.transforms(image=image_resized, 
                                     bboxes=target["boxes"],
                                     labels=target["labels"])
            image_resized = sample["image"]
            if len(sample['bboxes']) > 0:
                target['boxes'] = torch.Tensor(sample['bboxes']) 
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        image_resized = torch.as_tensor(image_resized, dtype=torch.float32)

        return image_resized, target
    
    def __len__(self):
        return self.csv_labels.shape[0]
    

# train_loader and valid_loader are used in the training loop

# Create the training dataset
def get_dataloaders(batch_size=BATCH_SIZE, resize_to=RESIZE_TO, num_workers=4):
    """
    Returns: train_loader, valid_loader
    """    
    train_dataset = PolypDataset(TRAIN_DIR, resize_to, resize_to, CLASSES, get_transform(train=True))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, 
        collate_fn=collate_fn
    )
    print(f"Train dataset size: {len(train_dataset)}")

    # Create the validation dataset
    valid_dataset = PolypDataset(VALID_DIR, resize_to, resize_to, CLASSES, get_transform(train=False))
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, 
        collate_fn=collate_fn
    )
    print(f"Valid dataset size: {len(valid_dataset)}\n")

    return train_loader, valid_loader


if __name__ == "__main__":
    dataset = PolypDataset(
        VALID_DIR, 
        RESIZE_TO, 
        RESIZE_TO, 
        CLASSES, 
        get_transform(train=False)
    )
    print(f"Test dataset size: {len(dataset)}\n")

    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = CLASSES[target["labels"]]
        if label != 'background':
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), 
                (int(box[2]), int(box[3])), (0, 255, 0), 1
            )
        cv2.putText(
            image, 
            label, 
            (int(box[0]), int(box[1]) - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        # TODO: switch to pyplot for image display to avoid cv2.imshow() bug
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)