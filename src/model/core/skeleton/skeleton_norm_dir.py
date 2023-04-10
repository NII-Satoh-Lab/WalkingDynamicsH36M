from typing import List

import numpy as np
import torch
import random

from model.utils.keypoints import center_kpts_around_hip, random_rotation_y_axis
from .skeleton import Skeleton

class SkeletonUnitNorm(Skeleton):
    node_hip = {0: 'NormHip', 1: 'DirHip'}
    hip_norm_treshold = 10000 # 10 m
        
       #################  functions to obtain input poses ########################################################################################
    
    def transform_seq_w_hip_to_input_space(self, data):
        centered, hips = data[..., 1:, :], data[..., 0, :].unsqueeze(-2)
        #set beginning of seq to zero
        # hips = hips - hips[...,0,:,:].unsqueeze(-3)
        hips = hips - hips[...,self.history_length-1,:,:].unsqueeze(-3)
        # compute residual
        
        # compute norm and unit vector
        hips_norm = torch.broadcast_to(torch.norm(hips, p='fro', dim=-1, keepdim=True), hips.shape) # [..., T, N, 3]
        # hips_norm = torch.abs(hips_norm) # cast to strictly positive, because sign is included in unit direction 
        hips = hips/hips_norm
        hips[hips_norm==0.] = 0.
        # hips_norm /= self.hip_norm_treshold
        assert torch.isnan(hips).sum() == 0
        assert torch.isnan(hips_norm).sum() == 0
        assert (hips_norm >= 0.).all() 
        assert (hips >= -1.).all()  and (hips <= 1.).all() 
        kpts = torch.cat([hips_norm, hips, centered], dim=-2)
        
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
    
    def transform_hip_to_metric_space(self, kpts):
        
        # hip_coords = kpts[...,1,:]*(kpts[...,0,:]*self.hip_norm_treshold) #norm*dir
        hip_coords = kpts[...,1,:]*kpts[...,0,:]
        hip_coords = hip_coords.unsqueeze(-2)
        return hip_coords

    
    ##########################  data augmentation function ##########################################################################
        
    def rescale_body(self, kpts):
        assert 0, "write me"
        kpts = kpts.clone()
        # shift_percentage = (random.random() - 0.5)*2*150 # in the range of +-15 cm
        # shift_hip = shift_percentage
        # shift_pose = shift_percentage/self.pose_box_size
        # def get_ratio(shift):
        #     return 1+ shift
        # kpts[..., 0, :] += shift_hip/kpts[..., 0, :] 
        # kpts[..., len(self.node_hip):, :] *= get_ratio(shift_pose)
        return kpts