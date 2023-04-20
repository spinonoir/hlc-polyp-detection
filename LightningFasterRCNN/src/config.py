from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os

"""
This file contains all of the configurations for the model.
"""

BATCH_SIZE = 32                                 # Number of images in each batch
RESIZE_TO = {'width': 800, 'height': 548}       # Resize images to this size
NUM_EPOCHS = 15                                 # Number of epochs to train for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths to training and validation data (jpf and xml files)
# TRAIN_DIR = os.path.join("data", "train")
# VALID_DIR = os.path.join("data", "val")
# TEST_DIR = os.path.join("data", "test")
# Defining our classes: for now just background and polyp
CLASSES = [
    "background",
    "polyp",
]
NUM_CLASSES = len(CLASSES)

# Do we output samples after transforming the images?
VISUALIZE_AFTER_TRANSFORM = False

# Directory for saving model checkpoints, plots, etc.
OUTPUT_DIR = os.path.join("output")
SAVE_PLOTS_EPOCH = 2 # Save plots every 2 epochs
SAVE_MODEL_EPOCH = 5 # Save model every 10 epochs