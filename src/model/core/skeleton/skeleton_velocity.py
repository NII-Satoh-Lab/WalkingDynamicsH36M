from typing import List

import numpy as np
import torch
import random

from model.utils.keypoints import center_kpts_around_hip, random_rotation_y_axis
from .skeleton import Skeleton

class SkeletonVelocity(Skeleton):
    residual_treshold = 600 # 1m, max motion of the hio between consecutive timesteps
    node_hip = {0: 'ResidualHip'}
    
    #################  functions to obtain input poses ########################################################################################
    
    def transform_seq_w_hip_to_input_space(self, data):
        centered, hips = data[..., 1:, :], data[..., 0, :].unsqueeze(-2)
        #set beginning of seq to zero
        hips = hips - hips[...,0,:,:].unsqueeze(-3)
        # compute residual
        hips = torch.cat([hips[...,0,:,:].unsqueeze(-3), hips[..., 1:,:,:] - hips[..., :-1, :,:]], dim=-3)
        hips /= self.residual_treshold

        assert torch.isnan(hips).sum() == 0
        assert (hips >= -1.).all()  and (hips <= 1.).all() 
        kpts = torch.cat([hips, centered], dim=-2)
        
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
    
    def transform_hip_to_metric_space(self, kpts):
        hip_coords = kpts[...,0,:].unsqueeze(-2)*self.residual_treshold
        hip_coords = torch.stack([hip_coords[..., 0, :, :]]+[hip_coords[..., :i+1, :,:].sum(dim=-3) for i in range(1,hip_coords.shape[-3])], dim=-3)# torch.cumsum(hip_coords, dim=0)
        return hip_coords
    
    ##########################  data augmentation function ##########################################################################
        
    def rescale_body(self, kpts):
        kpts = kpts.clone()
        shift_percentage = (random.random() - 0.5)*2*150 # in the range of +-15 cm
        shift_residual = shift_percentage/self.residual_treshold
        shift_pose = shift_percentage/self.pose_box_size
        def get_ratio(shift):
            return 1+ shift
        kpts[..., 0:len(self.node_hip), :] *= get_ratio(shift_residual)
        kpts[..., len(self.node_hip):, :] *= get_ratio(shift_pose)
        return kpts
