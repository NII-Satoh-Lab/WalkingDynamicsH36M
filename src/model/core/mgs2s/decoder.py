from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision

from model.core.layers.structural import StaticGraphLinear, StaticGraphGRU


class Decoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 feature_size: int,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 position: bool,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 dec_num_layers: int = 1,
                 dropout: float = 0.,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.position = position

        self.param_groups = param_groups
        # self.activation_fn = torch.nn.Tanh()
        self.num_layers = dec_num_layers
        self.if_consider_hip = True #kwargs["if_consider_hip"]
        
        if kwargs["hip_mode"] == "velocity":
            self.activation_fn = torch.nn.Tanh()
        elif kwargs["hip_mode"] == "norm":
            def activation(y_t_state):
                y_t_state[...,1:,:] = torch.tanh(y_t_state[...,1:,:])
                y_t_hip_norm = torch.broadcast_to(y_t_state[...,0,:].mean(-1).unsqueeze(-1), y_t_state[...,0,:].shape)
                y_t_state = torch.cat([y_t_hip_norm.unsqueeze(-2), y_t_state[...,1:,:]], dim=-2)
                return y_t_state
            self.activation_fn =activation
        else: 
            assert 0, "Not implemented"
            
        self.initial_hidden_h = StaticGraphLinear(latent_size + input_size + feature_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  learn_influence=True,
                                                  node_types=T)

        self.rnn = StaticGraphGRU(feature_size + latent_size + input_size,
                                  hidden_size,
                                  num_nodes=num_nodes,
                                  num_layers=dec_num_layers,
                                  learn_influence=True,
                                  node_types=T,
                                  recurrent_dropout=dropout,
                                  learn_additive_graph_influence=True,
                                  clockwork=False)

        self.fc_mean = StaticGraphLinear(hidden_size,
                                      output_size,
                                      num_nodes=num_nodes,
                                      learn_influence=True,
                                      node_types=T)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                ph: int = 1, state=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p_mean = list()

        x_t = x[:, -1]
        x_t_s = x_t.clone()
        if state is None:
            x_t_1 = x[:, -2] # we are teaking this one
        else:
            x_t_1 = state

        h_z = h   

        # Initialize hidden state of rnn
        rnn_h = self.initial_hidden_h(torch.cat([x_t_1, h_z], dim=-1))
        hidden = [(rnn_h, None)] * self.num_layers

        for i in range(ph):
            # Run RNN
            rnn_out, hidden = self.rnn(torch.cat([x_t, h_z], dim=-1).unsqueeze(1), hidden, i)  # [B, 1, N, D]
            y_t = rnn_out.squeeze(1)  # [B, N, D]
            y_t = self.dropout(self.activation_fn(y_t))

            y_t_state = self.fc_mean(y_t)
            y_t_state= self.activation_fn(y_t_state)
            
            p_mean.append(y_t_state)

        p_mean = torch.stack(p_mean, dim=1)

        return p_mean, x_t_s
