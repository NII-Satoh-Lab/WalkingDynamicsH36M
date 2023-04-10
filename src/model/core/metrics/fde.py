from typing import Tuple

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class FinalDisplacementError(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.y = None
        self.y_pred = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.y = None
        self.y_pred = None
        super().reset()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, _ = output

        if self.y is None:
            self.y = y[...,-1,:,:]
            self.y_pred = y_pred[...,-1,:,:]
        else:
            self.y = torch.cat([self.y, y[...,-1,:,:]], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred[...,-1,:,:]], dim=0)

    def compute(self):
        if self.y is None:
            raise NotComputableError('MeanPerJointPositionError must have at least one example before it can be computed.')
        fde = (self.y - self.y_pred).norm(dim=-1).mean()
        return fde