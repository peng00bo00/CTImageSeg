import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.classification import MulticlassAccuracy, MultilabelAveragePrecision, MulticlassJaccardIndex, MultilabelJaccardIndex
from torchmetrics.segmentation import MeanIoU


def MultiAcc(num_classes=10, average=None):
    """A wrapper of torchmetrics.classification.MulticlassAccuracy
    """

    def metric(y_pred, y_true):
        acc = MulticlassAccuracy(num_classes, average=average)
        return acc(y_pred, y_true)

    return metric

def MultiAP(num_classes=10, average=None):
    """A wrapper of torchmetrics.classification.MultilabelAveragePrecision
    """

    def metric(y_pred, y_true):
        # if y_pred.dim() != y_true.dim():
        #     y_true = y_true.squeeze()
        
        # if y_true.dim() == 1:
        #     y_true = F.one_hot(y_true, num_classes=num_classes)

        ap = MultilabelAveragePrecision(num_classes, average=average)
        return ap(y_pred, y_true)

    return metric

def MultiMIOU(num_classes=10):
    """A wrapper of torchmetrics.classification.MultilabelAveragePrecision
    """

    def metric(y_pred, y_true):
        # miou = MeanIoU(num_classes+1, include_background=False, per_class=True)
        miou = MultilabelJaccardIndex(num_labels=num_classes, average="none")
        return miou(y_pred, y_true)

    return metric