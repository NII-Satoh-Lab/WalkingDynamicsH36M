import numpy as np
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from typing import Tuple


class VIM(Metric):
    def __init__(self, output_transform=lambda x: x, dataset_name: str = "3dpw"):
        self.y = None
        self.y_pred = None
        self.mask = None
        self.dataset_name = dataset_name
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.y = None
        self.y_pred = None
        self.mask = None
        # self.visib = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        if len(output) == 2:
            y_pred, y = output
            mask = torch.ones(y[...,0].size(), device=y.device)
            # visib = torch.ones(input[...,0].size(), device=input.device)
        elif len(output) == 3:
            y_pred, y, mask = output
        else:
            y_pred, y, mask, _ = output

        if self.y is None:
            self.y = y
            self.y_pred = y_pred
            self.mask = mask
            # self.visib = visib
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
            self.mask = torch.cat([self.mask, mask], dim=0)
            # self.visib = torch.cat([self.visib, visib], dim=0)

    def compute(self):
        if self.y is None:
            raise NotComputableError('MeanAngleError must have at least one example before it can be computed.')
        b, f, k, d =self.y.shape
        self.y = self.y.view((b, f, k*d)).cpu().numpy()
        self.y_pred = self.y_pred.view((b, f, k*d)).cpu().numpy()
        self.mask = self.mask.cpu().numpy()
        vim = np.array([_VIM(y, y_pred, self.dataset_name, mask) 
               for y, y_pred, mask in zip(self.y, self.y_pred, self.mask)]) #B, T, K
        vim = torch.tensor(vim).mean(0)
        return vim

def _VIM(GT, pred, dataset_name, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        dataset_name: Dataset name
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:  (pred_len)
    """

    gt_i_global = np.copy(GT)

    if dataset_name == "posetrack":
        mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(gt_i_global - pred, 2) * mask
        #get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask,axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    else:   #3dpw
        errorPose = np.power(gt_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    return errorPose