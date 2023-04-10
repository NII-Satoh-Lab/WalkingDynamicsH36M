import torch
from typing import Tuple, Union, Optional
from model.core.mgs2s.mgs2s import MGS2S

class Motion(torch.nn.Module):
    def __init__(self,
                 num_nodes: int,
                 latent_size: int,
                 T: torch.Tensor = None, # joint types
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 **kwargs):

        super(Motion, self).__init__()

        self.param_groups = [{}]

        self._latent_size = latent_size

        num_nodes = num_nodes

        # Core Model
        self.core = MGS2S(num_nodes=num_nodes,
                          input_size=3,
                          G=G,
                          T=T,
                          param_groups=self.param_groups,
                          latent_size=latent_size,
                          **kwargs
                          )


    def forward(self, x: torch.Tensor, ph: int = 1, state=None) \
            -> Tuple[ torch.Tensor, Tuple, dict]:

        x0_state = x[:, [-1]].clone()

        if state is None:
            core_state = None
        else:
            core_state = state[0]

        x0 = x[:, -1].clone().unsqueeze(-2).repeat_interleave(repeats=self._latent_size, dim=-2)

        p_mean, core_state, kwargs = self.core(x, ph, core_state)
        
        # Permute from [B, T, N, D] to [B, T, N, D]
        # p_mean = p_mean.permute(0, 2, 3, 1, 4).contiguous()
        # p_mean = p_mean[:,:,:,0,:]
        state = (core_state, x0_state)
        return p_mean, state, {**kwargs}

    # def loss(self, y_pred, y, **kwargs):
    #     return self.core.loss(y_pred, y, **kwargs)
    
    def loss_pose(self, pred, y, **kwargs):
        pred, y = pred[...,1:,:], y[...,1:,:] #centered coords
        return self.core.loss(pred, y, type="mse", **kwargs)
    
    def loss_hip(self, pred, y, **kwargs):
        # hip norm
        pred_hip, y_hip = pred[...,0,:].unsqueeze(-2), y[...,0,:].unsqueeze(-2)
        return self.core.loss(pred_hip, y_hip, **kwargs)
