import numpy as np
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from typing import Tuple


class VAM(Metric):
    def __init__(self, output_transform=lambda x: x, occ_cutoff: float = 200):
        self.y = None
        self.y_pred = None
        self.pred_visib = None
        self.occ_cutoff = occ_cutoff
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.y = None
        self.y_pred = None
        self.pred_visib = None
        # self.visib = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        if len(output) == 2:
            y_pred, y = output
            pred_visib = torch.ones(y[...,0].size(), device=y.device)
            # visib = torch.ones(input[...,0].size(), device=input.device)
        elif len(output) == 3:
            y_pred, y, pred_visib = output
        else:
            y_pred, y, pred_visib, _ = output

        if self.y is None:
            self.y = y
            self.y_pred = y_pred
            self.pred_visib = pred_visib
            # self.visib = visib
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
            self.pred_visib = torch.cat([self.pred_visib, pred_visib], dim=0)
            # self.visib = torch.cat([self.visib, visib], dim=0)

    def compute(self):
        if self.y is None:
            raise NotComputableError('MeanAngleError must have at least one example before it can be computed.')
        b, f, k, d =self.y.shape
        self.y = self.y.view((b, f, k*d)).cpu().numpy()*1000
        self.y_pred = self.y_pred.view((b, f, k*d)).cpu().numpy()*1000
        self.pred_visib = self.pred_visib.cpu().numpy()
        vam = [_VAM(y, y_pred, self.occ_cutoff, pred_visib) 
               for y, y_pred, pred_visib in zip(self.y, self.y_pred, self.pred_visib)] #B, T, K
        vam = torch.stack(vam,dim=0).mean(0)
        return vam



def _VAM(GT, pred, occ_cutoff, pred_visib):
    """
    Visibility Aware Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        occ_cutoff: Maximum error penalty
        pred_visib: Predicted visibilities of pose, array of shape (pred_len, #joint)
    Output:
        seq_err:
    """
    pred_visib = np.repeat(pred_visib, 2, axis=-1)
    # F = 0
    seq_err = []
    if type(GT) is list:
        GT = np.array(GT)
    GT_mask = np.where(abs(GT) < 0.5, 0, 1)

    for frame in range(GT.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, GT.shape[1], 2):
            if GT_mask[frame][j] == 0:
                if pred_visib[frame][j] == 0:
                    dist = 0
                elif pred_visib[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif GT_mask[frame][j] > 0:
                N += 1
                if pred_visib[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_visib[frame][j] == 1:
                    d = np.power(GT[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            f_err += dist
        
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
        # if f_err > 0:
        # F += 1
    return np.array(seq_err)