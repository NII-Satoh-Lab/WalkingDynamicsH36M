from typing import List

import numpy as np
import torch
import random

# from model.utils.keypoints import center_kpts_around_hip
from model.core.skeleton.skeleton import Skeleton
from model.core.skeleton.skeleton_velocity import SkeletonVelocity 
from model.core.skeleton.skeleton_norm_dir import SkeletonUnitNorm

def add_center_hip(kpts, hip_idxs: list):
    center = kpts[..., hip_idxs[0], :] + (kpts[..., hip_idxs[1], :]-kpts[..., hip_idxs[0], :])/2    
    kpts = torch.cat([center.unsqueeze(-2), kpts], dim=-2)
    return kpts


class SoMoFSkeleton(Skeleton):
    residual_treshold = 600 # 1m, max motion of the hio between consecutive timesteps

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.joint_dict_orig  = {0: 'LHip', 1: 'RHip',
                     2: 'LKnee',  3: 'RKnee', 4: 'LAnkle', 5: 'RAnkle',
                     6: 'Head',
                     7: 'LShoulder', 8: 'RShoulder',
                     9: 'LElbow', 10: 'RElbow', 11: 'LWrist', 12: 'RWrist'}
        self.limbseq = [[0, 1], [0,2], [2,4],
                        [1,3], [3,5],
                        [0,7], [7,9], [9,11],
                        [7,8], [6,7], [6,8],
                        [1,8], [8,10], [10,12],
                        ]
        assert self.num_joints ==13

        if self.if_consider_hip:
            self.node_dict = {**self.node_hip, **{i +len(self.node_hip): v for i,v in self.joint_dict_orig.items()}}
        else: 
            self.node_dict =  self.joint_dict_orig
        self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]

    #################  functions to obtain input poses ########################################################################################
    
    def tranform_to_input_space_pose_only(self, data):
        data = add_center_hip(data, hip_idxs=[0,1])
        return super().tranform_to_input_space_pose_only(data)


    
class SoMoFSkeletonVelocity(SoMoFSkeleton, SkeletonVelocity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        
class SoMoFSkeletonUnitNorm(SoMoFSkeleton, SkeletonUnitNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)