import torch
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert

from functools import wraps
import gc

from torchvision.ops import box_iou


def flush_and_gc(f):
  @wraps(f)
  def g(*args, **kwargs):
    torch.cuda.empty_cache()
    gc.collect()
    return f(*args, **kwargs)
  return g

# Main Lightning Module for FasterRCNN 
# With additional pyra
class LightningFasterModule(pl.LightningModule):
    def __init__(self,config=None):
        super().__init__()
        assert config is not None, "Config is required"
        self.config = config
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features 
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, config['num_classes'])


    # def validation_step(self, batch, batch_idx):
    #     images, targets = batch
    #     # fasterrcnn takes only images for eval() mode
    #     outs = self.model(images)
    #     ious = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)])
    #     iou = ious.mean()
    #     metrics = _evaluate_metrics(iou, self.thresholds)
    #     self.log("val_iou", iou, prog_bar=True)
    #     return {"val_iou": iou, **metrics}
    
    # def validation_epoch_end(self, outputs):
    #     val_iou = torch.stack([x["val_iou"] for x in outputs]).mean()
    #     val_metrics = {k: torch.stack([x[k] for x in outputs]).mean() for k in outputs[0].keys() if k != "val_iou"}
    #     report_thresh = self.thresholds[-1]
    #     prec, rec, f1 = val_metrics[f"precision_{report_thresh}"], val_metrics[f"recall_{report_thresh}"], val_metrics[f"f1_{report_thresh}"]
    #     self.log("val_iou", val_iou, f'recall_{report_thresh}', rec, f'prec_{report_thresh}', prec, f'f1_{report_thresh}', f1 , prog_bar=True)
    #     return {"val_iou": val_iou, **val_metrics}

    def full_train(self):
        self.detector.requires_grad = True

    def forward(self, x):
        self.detector.eval()
        return self.detector(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.config['lr'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay'],
            nesterov=self.config['nesterov']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=3, 
            T_mult=1,
            eta_min=self.config['min_lr'],
            verbose=True
        )
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "monitor": "val_loss" }}
    

    # @flush_and_gc
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # print(f"{type(targets)=} {type(targets[0])=} {type(targets[0]['labels'])=}")
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.detector(images, targets)

        pred_boxes = self.forward(images)
        self.detector.train()
        # print(f"{type(targets)=} {len(pred_boxes)=}")
        # print(f"{targets=}")
        # print(f"{pred_boxes=}")
        # for i in range(len(pred_boxes)):
        #     print(f"{type(pred_boxes[i])=}")
        

        loss_dict['loss_recall'] = 1 - torch.mean(torch.stack([self.recall(b["boxes"], pb["boxes"], iou_threshold=0.5) for b, pb in zip(targets, pred_boxes)]))
        loss = sum(loss_dict.values())

        loss_dict = {k:(v.detach() if hasattr(v, 'detach') else v) for k, v in loss_dict.items()}
        self.log("loss", loss, batch_size=self.config['batch_size'])
        self.log_dict(loss_dict)
        return {"loss": loss, "log": loss_dict}

    # @flush_and_gc
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        pred_boxes = self.forward(images)

        # print("IN VALIDATION STEP")
        # print(f"{type(targets)=} {type(pred_boxes)=}")
        # print(f"{pred_boxes=}")
        # print(f"{pred_boxes=}")
        # for i in range(len(pred_boxes)):
        #     print(f"{type(pred_boxes[i])=}")

        recall_list = [self.recall(b["boxes"], pb["boxes"], iou_threshold=0.5) for b, pb in zip(targets, pred_boxes)]
        self.val_recall = torch.mean(torch.stack(recall_list))
        # self.val_recall = torch.mean(torch.stack([self.recall(b["boxes"], pb["boxes"], iou_threshold=0.5) for b, pb in zip(targets, pred_boxes[0])]))
        self.log("val_recall", self.val_recall, batch_size=self.config['batch_size'])
    
        return self.val_recall

    # @flush_and_gc
    def test_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        pred_boxes = self.forward(images)

        self.test_recall = torch.mean(torch.stack([self.recall(b["boxes"], pb["boxes"], iou_threshold=0.5) for b, pb in zip(targets, pred_boxes)]))
        self.log("test_recall", self.test_recall, batch_size=self.config['batch_size'])

        return self.test_recall


    def recall(self, target_boxes, pred_boxes, iou_threshold=1.0):
        """
        Calculate the recall for a given target and prediction boxes
        returns the recall
        """
        total_gt = len(target_boxes)
        total_pred = len(pred_boxes)

        # if there are boxes in both target and pred, calculate the scores
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on the iou
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_qual_matrix = box_iou(target_boxes, pred_boxes)

            results = matcher(match_qual_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            # matched_elements = results[results != -1]

            # Matcher can only match a pred twice
            # false_positive = torch.count_nonzero(matched_elements == -1) + (len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive

            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else torch.tensor(0.0, device=pred_boxes.device)

        
        elif total_gt == 0:
            if total_pred > 0:
                recall = torch.tensor(0.0, device=pred_boxes.device)

            else:
                recall = torch.tensor(1.0, device=pred_boxes.device)

            
        elif total_gt > 0 and total_pred == 0:
            recall = torch.tensor(0.0, device=pred_boxes.device)


        return recall
        

    def evaluate_metrics(self, target_boxes, pred_boxes, iou_threshold=1.0):
        """
        Calculate precision, accuracy, f1 and recall for a given target and prediction boxes
        Store in self.metric_dict
        returns the recall
        """
        total_gt = len(target_boxes)
        total_pred = len(pred_boxes)

        # if there are boxes in both target and pred, calculate the scores
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on the iou
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_qual_matrix = box_iou(target_boxes, pred_boxes)

            results = matcher(match_qual_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results != -1]

            # Matcher can only match a pred twice
            false_positive = torch.count_nonzero(matched_elements == -1) + (len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive

            accuracy = true_positive / (true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative > 0 else 0
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            # self.metric_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        
        elif total_gt == 0:
            if total_pred > 0:
                accuracy = torch.tensor(0.0, device=pred_boxes.device)
                precision = torch.tensor(0.0, device=pred_boxes.device)
                recall = torch.tensor(0.0, device=pred_boxes.device)
                f1 = torch.tensor(0.0, device=pred_boxes.device)
                # self.metric_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

            else:
                accuracy = torch.tensor(1.0, device=pred_boxes.device)
                precision = torch.tensor(1.0, device=pred_boxes.device)
                recall = torch.tensor(1.0, device=pred_boxes.device)
                f1 = torch.tensor(1.0, device=pred_boxes.device)
                # self.metric_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

            
        elif total_gt > 0 and total_pred == 0:
            accuracy = torch.tensor(0.0, device=pred_boxes.device)
            precision = torch.tensor(0.0, device=pred_boxes.device)
            recall = torch.tensor(0.0, device=pred_boxes.device)
            f1 = torch.tensor(0.0, device=pred_boxes.device)
            # self.metric_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


    # def validation_epoch_end(self, outs):
    #     avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
    #     logs = {"val_iou": avg_iou}
    #     return {"avg_val_iou": avg_iou, "log": logs}

    # def configure_optimizers(self):
    #     return torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=self.learning_rate,
    #         momentum=0.9,
    #         weight_decay=0.005,
    #     )
    
