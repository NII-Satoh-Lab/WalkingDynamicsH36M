from typing import Tuple

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class Difference(Metric):
    def __init__(self, output_transform=lambda x: x, keep_time_dim: bool = True, keep_joint_dim: bool = False):
        self.y = None
        self.y_pred = None
        self.keep_time_dim = keep_time_dim
        self.keep_joint_dim =  keep_joint_dim
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
            self.y = y
            self.y_pred = y_pred
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)

    def compute(self):
        if self.y is None:
            raise NotComputableError('MeanPerJointPositionError must have at least one example before it can be computed.')
        ret = torch.abs(self.y - self.y_pred).mean(dim=[0])
        if not self.keep_joint_dim:
            ret = ret.mean(dim=[-1])
        if not self.keep_time_dim:
            ret = ret.mean(0)
        return ret
