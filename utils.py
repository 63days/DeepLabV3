import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CenterCrop(object):
    def __call__(self, x, S):
        W = x.size(2)
        off = W - S
        start = math.ceil(off/2)
        end = math.floor(off/2)

        # 3-> 1, 2 4-> 2, 2  x-> x/2 버림 x/2올림
        return x[:, :, start:-end, start:-end]

class Padding(object):
    def __call__(self, x, S):
        W = x.size(3)
        diff = S - W
        start = math.ceil(diff / 2)
        end = math.floor(diff / 2)
        padding = (start, end, start, end)
        return F.pad(x, padding, mode='constant', value=0)



def get_IOU(pred, label, threshold=0.5, eps=1e-5):
    # TP-> pred: True  label: True
    # FN-> pred: False label: True
    # FP-> pred: True  label: False
    pred = (pred > threshold).long()

    pred_True = (pred == 1)
    pred_False = ~pred_True

    label_True = (label == 1)
    label_False = ~label_True

    TP = torch.logical_and(pred_True, label_True).float().sum()
    FN = torch.logical_and(pred_False, label_True).float().sum()
    FP = torch.logical_and(pred_True, label_False).float().sum()

    IOU = TP / (TP + FN + FP)

    return IOU.item()