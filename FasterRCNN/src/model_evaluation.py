from torchvision.ops import boxes as box_ops
import torch

def evaluate(model, data_loader, device, iou_threshold=0.5):
    pass

def get_iou_types(model):
    iou_types = ["bbox"]
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# Calculate the intersection over union (IoU) of two sets of bounding boxes.
def generalized_box_iou(boxes1, boxes2):
    # boxes1: (N, 4)
    # boxes2: (M, 4)
    # Return: (N, M)
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]
    wh = (rb - lt + 1).clamp(min=0) # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N,M]
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

# Calulate the recall, precision, and f1 score of the model.
def calculate_metrics(iou, threshold):
    pass