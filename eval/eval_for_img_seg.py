import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from input.dataset_config import CITYSCAPES
from tqdm import tqdm
import cv2
import torch.nn as nn


class AccEval(object):

    def __init__(self):
        self.score_list = []

    def __call__(self, pred,target):
        total_element = 1
        for i in pred.size():
            total_element *= i
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        not_equal = torch.sum(torch.abs(pred.float() - target.float())).data
        score = 1 - float(not_equal) / float(total_element)
        self.score_list.append(score)
        return score

    def get_meanscore(self):
        return np.nanmean(self.score_list)



class DiceEval(object):

    def __init__(self):
        self.score_list = []

    def __call__(self, pred,target):
        eps = 1e-5
        common = torch.dot(pred.contiguous().view(-1).float(), target.contiguous().view(-1).float()).data
        union = torch.sum(pred.view(-1)).data + torch.sum(target.view(-1)).data
        score = (float(2 * common) + eps) / (float(union) + eps)
        self.score_list.append(score)
        return score

    def get_meanscore(self):
        return np.nanmean(self.score_list)

class IoUEval(object):

    def __init__(self):
        self.score_list = []

    def __call__(self, pred,target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        tp_idxs = (pred == 1)
        cls_idxs = (target == 1)

        intersection = (tp_idxs[cls_idxs]).long().sum().data
        union = tp_idxs.long().sum().data + cls_idxs.long().sum().data - intersection
        eps = 1e-5
        score = (float(intersection) + eps) / (float(union) + eps)
        self.score_list.append(score)
        return score

    def get_meanscore(self):
        return np.nanmean(self.score_list)


class MIoU(object):

    def __init__(self,num_cls,ignore=255):

        assert isinstance(num_cls,int) and num_cls>=1
        self.num_cls = num_cls
        self.confusion_matrix = np.zeros(shape=(num_cls,num_cls))
        self.ignore = ignore

    def _hist(self, pred, label):
        mask = (label!=self.ignore)
        hist = np.bincount(self.num_cls * label[mask].astype(int) + pred[mask],
                           minlength=self.num_cls ** 2).reshape(self.num_cls, self.num_cls)
        return hist

    def __call__(self, pred,label):
        pred = pred.view(-1).cpu().numpy()
        label = label.view(-1).cpu().numpy()
        hist = self._hist(pred, label)
        self.confusion_matrix += hist


    def miou(self):
        hist = self.confusion_matrix
        intersection = np.diag(hist)
        # + np.finfo(np.float32).eps
        iou = intersection / (hist.sum(axis=1) + hist.sum(axis=0) - intersection )
        miou = np.nanmean(iou)
        return iou,miou


