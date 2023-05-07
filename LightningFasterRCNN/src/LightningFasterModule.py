import torch
import torchvision
import pytorch_lightning as L
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.fcos import FCOS, FCOSHead
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
class LightningFasterModule(L.LightningModule):
    def __init__(self,config=None):
        super().__init__()
        assert config is not None, "Config is required"
        self.config = config
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", trainable_backbone_layers=5)

        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features 
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, config['num_classes'])
        self.callback_metrics = { 
            'test': {
                'test_acc': [],
                'test_precision': [],
                'test_recall': [],
                'test_f1': []
                },
            'val': {
                'val_acc': [],
                'val_precision': [],
                'val_recall': [],
                'val_f1': []
            }
        }


        self.save_hyperparameters(config)


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

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        

        self.log("train_loss", losses, batch_size=self.config['batch_size'])
        self.log_dict(loss_dict, batch_size=self.config['batch_size'])

        return {"loss": losses, "log": loss_dict}

    # @flush_and_gc
    def validation_step(self, batch, batch_idx):

        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # Inferece
        pred_boxes = self.detector(images)
        stacked_metrics = torch.stack([self.evaluate_metrics(t["boxes"], o["boxes"]) for t, o in zip(targets, pred_boxes)])
        stacked_metrics = stacked_metrics.mean(dim=0)
        acc, prec, rec, f1 = stacked_metrics

        log_dict = {"val_acc": acc, "val_precision": prec, "val_recall": rec, "val_f1": f1}

        self.callback_metrics['val']['val_acc'].append(acc)
        self.callback_metrics['val']['val_precision'].append(prec)
        self.callback_metrics['val']['val_recall'].append(rec)
        self.callback_metrics['val']['val_f1'].append(f1)


        self.log_dict(log_dict, batch_size=self.config['batch_size'])


        self.log("val_acc", acc, batch_size=self.config['batch_size'])
        self.log("val_precision", prec, batch_size=self.config['batch_size'])
        self.log("val_recall", rec, batch_size=self.config['batch_size'])
        self.log("val_f1", f1, batch_size=self.config['batch_size'])

    
        return log_dict
    

    # @flush_and_gc
    def test_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        # Inferece
        pred_boxes = self.detector(images)
        stacked_metrics = torch.stack([self.evaluate_metrics(t["boxes"], o["boxes"]) for t, o in zip(targets, pred_boxes)])
        stacked_metrics = stacked_metrics.mean(dim=0)
        acc, prec, rec, f1 = stacked_metrics

        self.callback_metrics['test']['test_acc'].append(acc)
        self.callback_metrics['test']['test_precision'].append(prec)
        self.callback_metrics['test']['test_recall'].append(rec)
        self.callback_metrics['test']['test_f1'].append(f1)

        log_dict = {"test_acc": acc, "test_precision": prec, "test_recall": rec, "test_f1": f1}
        self.log_dict(log_dict, batch_size=self.config['batch_size'])


        self.log("test_acc", acc, batch_size=self.config['batch_size'])
        self.log("test_precision", prec, batch_size=self.config['batch_size'])
        self.log("test_recall", rec, batch_size=self.config['batch_size'])
        self.log("test_f1", f1, batch_size=self.config['batch_size'])

        return log_dict


    def on_test_epoch_end(self):
        precisions = torch.as_tensor(self.callback_metrics['test']['test_precision'], dtype=torch.float32)
        recalls = torch.as_tensor(self.callback_metrics['test']['test_recall'], dtype=torch.float32)
        f1s = torch.as_tensor(self.callback_metrics['test']['test_f1'], dtype=torch.float32)

        max_precision = precisions.max()
        max_recall = recalls.max()
        max_f1 = f1s.max()

        min_precision = precisions.min()
        min_recall = recalls.min()
        min_f1 = f1s.min()

        avg_precision = precisions.mean()
        avg_recall = recalls.mean()
        avg_f1 = f1s.mean()
            
        
        self.log_dict({
            "test_max_precision": max_precision,
            "test_max_recall": max_recall,
            "test_max_f1": max_f1,
            "test_min_precision": min_precision,
            "test_min_recall": min_recall,
            "test_min_f1": min_f1,
            "test_avg_precision": avg_precision,
            "test_avg_recall": avg_recall,
            "test_avg_f1": avg_f1,
        })

        for key in self.callback_metrics['test'].keys():
            self.callback_metrics['test'][key] = []

    def on_validation_epoch_end(self):
        precisions = torch.as_tensor(self.callback_metrics['val']['val_precision'], dtype=torch.float32)
        recalls = torch.as_tensor(self.callback_metrics['val']['val_recall'], dtype=torch.float32)
        f1s = torch.as_tensor(self.callback_metrics['val']['val_f1'], dtype=torch.float32)
        
        
        max_precision = precisions.max()
        max_recall = recalls.max()
        max_f1 = f1s.max()

        min_precision = precisions.min()
        min_recall = recalls.min()
        min_f1 = f1s.min()

        avg_precision = precisions.mean()
        avg_recall = recalls.mean()
        avg_f1 = f1s.mean()

        
        self.log_dict({
            "val_max_precision": max_precision,
            "val_max_recall": max_recall,
            "val_max_f1": max_f1,
            "val_min_precision": min_precision,
            "val_min_recall": min_recall,
            "val_min_f1": min_f1,
            "val_avg_precision": avg_precision,
            "val_avg_recall": avg_recall,
            "val_avg_f1": avg_f1,
        })

        for key in self.callback_metrics['val'].keys():
            self.callback_metrics['val'][key] = []

        



    def evaluate_metrics(self, target_boxes, pred_boxes, iou_threshold=0.5):
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


        
        elif total_gt == 0:
            if total_pred > 0:
                accuracy = torch.tensor(0.0, device=pred_boxes.device)
                precision = torch.tensor(0.0, device=pred_boxes.device)
                recall = torch.tensor(0.0, device=pred_boxes.device)
                f1 = torch.tensor(0.0, device=pred_boxes.device)


            else:
                accuracy = torch.tensor(1.0, device=pred_boxes.device)
                precision = torch.tensor(1.0, device=pred_boxes.device)
                recall = torch.tensor(1.0, device=pred_boxes.device)
                f1 = torch.tensor(1.0, device=pred_boxes.device)


            
        elif total_gt > 0 and total_pred == 0:
            accuracy = torch.tensor(0.0, device=pred_boxes.device)
            precision = torch.tensor(0.0, device=pred_boxes.device)
            recall = torch.tensor(0.0, device=pred_boxes.device)
            f1 = torch.tensor(0.0, device=pred_boxes.device)


        return torch.as_tensor([accuracy, precision, recall, f1], device=pred_boxes.device)