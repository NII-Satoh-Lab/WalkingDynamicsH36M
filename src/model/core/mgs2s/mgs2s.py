from typing import Union

import torch
import torch.nn as nn

from model.core.mgs2s.decoder import Decoder
from model.core.mgs2s.encoder import Encoder

class MGS2S(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 encoder_hidden_size: int,
                 bottleneck_size: int,
                 decoder_hidden_size: int,
                 output_size: int,
                 position: bool = False,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups

        self._num_nodes = num_nodes
        self._position = position

        self.encoder = Encoder(num_nodes=num_nodes,
                               input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               output_size=bottleneck_size,
                               G=G,
                               T=T,
                               **kwargs)

        self.decoder = Decoder( num_nodes=num_nodes,
                               input_size=bottleneck_size,
                               feature_size=input_size,
                               hidden_size=decoder_hidden_size,
                               output_size=output_size,
                               position=position,
                               G=G,
                               T=T,
                               param_groups=self.param_groups,
                               **kwargs
                               )

        self.z_dropout = nn.Dropout(kwargs['dropout'])
        self.act_funct = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, ph=1, state=None):
        bs = x.shape[0] # B, T, N, 3

        if state is None:
            state = (None, None)

        # Encode History
        h, encoder_state = self.encoder(x, state[0])  # [B, N, D] num_joints, dimension

        x_tiled = x[:, -2:]
        x_t_tiled = x[:, -1]

        # Decode future q
        p_mean, decoder_state = self.decoder(x=x_tiled,
                                            h=h,
                                            ph=ph,
                                            state=state[1])  # [B, T, N, D]

        return p_mean, (encoder_state, decoder_state), {}
    
    def loss(self, pred, y, type="mse"):  
        if type=="mse":
            out = torch.nn.MSELoss(reduction="none")(pred,y)
        elif type in ["l1", "L1"]:
            out = torch.nn.L1Loss(reduction="none")(pred,y)
        else: 
            assert 0, "Not implemnted"
        loss = (out.sum(-1) #spatial size [32, 13, 25, 5, 128], 
                                            .mean(-1) #keypoints
                                            .mean(-1) # timesteps
                                            .mean())
        return loss
    
    def loss_ohkm(self, pred, y, type="mse"):  
        if type=="mse":
            out = torch.nn.MSELoss(reduction="none")(pred,y)
        elif type in ["l1", "L1"]:
            out = torch.nn.L1Loss(reduction="none")(pred,y)
        else: 
            assert 0, "Not implemnted"
        loss = out.sum(-1)
        sorted_loss, __ = torch.sort(loss, dim=- 1, descending=True, stable=True)
        loss = (sorted_loss[..., :6].mean(-1) #keypoints
                                            .mean(-1) # timesteps
                                            .mean())
        return loss
     
