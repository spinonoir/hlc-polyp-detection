from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import albumentations as A
# import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2

from src.config import DEVICE, CLASSES as classes 

class Averager:
    """
    Averager class to keep track of training and validation loss,
    and to get the average for each epoch.
    """
    def __init__(self):
        self.curr_total = 0.0
        self.iters = 0.0


    def send(self, value):
        self.curr_total += value
        self.iters += 1

    @property
    def value(self):
        val = 0 if self.iters == 0 else 1.0 * self.curr_total / self.iters
        return val
    
    def reset(self):
        self.curr_total = 0.0
        self.iters = 0.0



def collate_fn(batch):
    """
    Custom collate function to be used in the DataLoader.
    This function will come in handy when we start loading in data
    where there is more than one bounding box per image.
    """
    return tuple(zip(*batch))


def get_transform(train=True):
    """
    Get the transform to be used for training or testing.
    Transformations from Albumentations are applied to the images 
    and targets and gives us some data augmentation.
    """
    # 

    # The bbox_params for Albumentations. We've included min_area to ensure that 
    # albumentations skips over any negative (background) images (no polyps) in 
    # our dataset.
    bbox_p = A.BboxParams(
        format='pascal_voc',
        min_visibility=0.1,
        min_area=128, 
        label_fields=['labels'])
    
    if train:
        return A.Compose(
                [
                    # TODO: Add more transformations and see what works best.
                    A.Flip(0.5),
                    A.RandomRotate90(0.5),
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ToTensorV2(p=1.0),
                ], 
                bbox_params=bbox_p,
            )

    else: 
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params=bbox_p)
    

# def show_transformed_image(train_loader):
#     """
#     Note that openCV's imshow is trash and forces you to close
#     the window by pressing a key on your keyboard. If you close 
#     the window manually everything will crash. 
#     This functiuon can be used to verify that the the transformed 
#     image label pairs in the data loader are correct. (For debugging)
#          [When VISUALIZE_AFTER_TRANSFORM is set to True in config.py]
#     """
#     print("Press any key to close the stupid openCV window. (I'm sorry.")
#     if len(train_loader) > 0:
#         for i in range(1): # Just show one image
#             images, targets = next(iter(train_loader))
#             images = list(image.to(DEVICE) for image in images)
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#             boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#             sample = images[i].permute(1, 2, 0).cpu().numpy()
#             # Prevent cv2 from attempting to draw a bbox on negative data (background)
#             # TODO: Modify this so that it can iterate though all the the labels.
#             # although, only the first label will ever say background in the training set
#             # but who knows what the model will do. 
#             if targets[i]['labels'][0] is not classes.index('background'):
#                 for box in boxes:
#                     cv2.rectangle(sample, 
#                                 (box[0], box[1]), 
#                                 (box[2], box[3]), 
#                                 (0, 0, 255), 2)
#             # TODO: switch to pyplot for displayig image previews. 
#             cv2.imshow('Transformed Image', sample)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()