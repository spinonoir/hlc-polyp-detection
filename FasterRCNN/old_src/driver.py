from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.PolypDataset import PolypDataset
from src.config import *
from src.utils import *

import torch
from torch.utils.data import Dataset, DataLoader

# This driver file only exists for debugging purposes.
def main():
    polypdataset = PolypDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_transform(train=True))
    dataloader = DataLoader(polypdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    images, targets = next(iter(dataloader))

    print(images[0].shape)

    show_transformed_image(dataloader)

if __name__ == '__main__':
    main()