from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    """
    Create a model using torchvision's fasterrcnn_resnet50_fpn model.
    :param num_classes: Number of classes in the new dataset

    Note this code is boilerplate and copilot auto-generated everything after
     def create_model(num_classes): lol
    """
    # Load a model pre-trained 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
