import torch
from typing import Tuple, Union, Optional
from model.core.mgs2s.mgs2s import MGS2S

class Motion(torch.nn.Module):
    def __init__(self,
                 num_nodes: int,
                 T: torch.Tensor = None, # joint types
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 **kwargs):

        super(Motion, self).__init__()

        self.param_groups = [{}]

        num_nodes = num_nodes

        # Core Model
        self.core = MGS2S(num_nodes=num_nodes,
                          input_size=3,
                          G=G,
                          T=T,
                          param_groups=self.param_groups,
                          **kwargs
                          )


    def forward(self, x: torch.Tensor, ph: int = 1, state=None) \
            -> Tuple[ torch.Tensor, Tuple, dict]:

        x0_state = x[:, [-1]].clone()

        if state is None:
            core_state = None
        else:
            core_state = state[0]

        p_mean, core_state, kwargs = self.core(x, ph, core_state)
        
        state = (core_state, x0_state)
        return p_mean, state, {**kwargs}
    
    def loss_pose(self, pred, y, **kwargs):
        pred, y = pred[...,1:,:], y[...,1:,:] #centered coords
        return self.core.loss(pred, y, type="mse", **kwargs)
    
    def loss_hip(self, pred, y, **kwargs):
        # hip norm
        pred_hip, y_hip = pred[...,0,:].unsqueeze(-2), y[...,0,:].unsqueeze(-2)
        return self.core.loss(pred_hip, y_hip, **kwargs)
