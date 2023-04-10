from typing import List

import numpy as np
import torch
from model.core.skeleton.skeleton import Skeleton
from model.core.skeleton.skeleton_velocity import SkeletonVelocity 
from model.core.skeleton.skeleton_norm_dir import SkeletonUnitNorm

class H36MSkeleton(Skeleton):
    residual_treshold = 600 # 1m, max motion of the hio between consecutive timesteps

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        if kwargs["num_joints"] == 25:
            self.joint_dict_orig = {0: "Hip",
                        1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RFoot", 5: "RToes",
                        6: "LHip", 7: "LKnee", 8: "LAnkle", 9: "LFoot", 10: "LToes",
                        11: "Torso", 12: "Neck", 13: "Nose", 14: "Head",
                        15: "LShoulder", 16: "LElbow", 17: "LWrist",
                        18: "LSmallFinger", 19: "LThumb",
                        20: "RShoulder", 21: "RElbow", 22: "RWrist",
                        23: "RSmallFinger", 24: "RThumb"}
            self.limbseq = [[0, 1], [0,6], [0,11],
                    [1,2], [2,3], [3,4], [4,5],# right foot
                    [6,7], [7,8], [8,9], [9,10],# left foot
                    [11,12], [12,13], [13,14], # head
                    [12, 15], [12,20],
                    [15,16], [16,17], [17,18], [17,19], # left hand
                    [20,21], [21,22], [22,23], [22,24] # right hand
                    ] 
            self.limbseq_names = ['Hip2RHip',
                    'Hip2LHip',
                    'Hip2Torso',
                    'RHip2RKnee',
                    'RKnee2RAnkle',
                    'RAnkle2RFoot',
                    'RFoot2RToes',
                    'LHip2LKnee',
                    'LKnee2LAnkle',
                    'LAnkle2LFoot',
                    'LFoot2LToes',
                    'Torso2Neck',
                    'Neck2Nose',
                    'Nose2Head',
                    'Neck2LShoulder',
                    'Neck2RShoulder',
                    'LShoulder2LElbow',
                    'LElbow2LWrist',
                    'LWrist2LSmallFinger',
                    'LWrist2LThumb',
                    'RShoulder2RElbow',
                    'RElbow2RWrist',
                    'RWrist2RSmallFinger',
                    'RWrist2RThumb']
        elif kwargs["num_joints"] == 17:
            self.joint_dict_orig = {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle",
                       4: "RFoot ", 5: "LHip", 6: " LKnee",
                       7: "LAnkle", 8: "LFoot", 9: "Neck", 10: "Head",
                       11: "LShoulder", 12: "LElbow", 13: "LWrist",
                       14: "RShoulder", 15: "RElbow", 16: "RWrist"}
            
            self.limbseq = [[0, 1], [0,5],
                   [1,2], [2,3], [3,4], # right foot
                   [5,6], [6,7], [7,8],# left foot
                   [0,9], [9,10], [9,11], [9,14], # head
                   [11,12], [12,13], # left hand
                    [14,15], [15,16] # right hand
                   ]
        else:
            assert 0, "Not implemented"
        if not self.if_consider_hip:
                self.node_dict = self.joint_dict_orig.copy()
                self.node_dict.pop(0)
                self.node_dict = {i:v for i,v in enumerate(list(self.node_dict.values()))}
        else: 
            self.node_dict = {k:v for k,v in enumerate(list(self.node_hip.values())+list(self.joint_dict_orig.values())[1:])}

        self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]


##########################  function to tranform back to metric space ##########################################################################
    
    def transform_to_metric_space(self, kpts):
        """_summary_

        Args:
            kpts (tensor (T,K,3)): coords of kpts in range [-1,1] in coordinate system that has hip as center. hips not included
            hip_coords (tensor (T,1,3)): hip position in camera coordinates

        Returns:
            kpts_cam: kpts in camera coordinates
        """

        if self.if_consider_hip:
            hip_coords = self.transform_hip_to_metric_space(kpts) 
            
            kpts = self.rescale_to_hip_box(kpts[...,len(self.node_hip):,:])
            kpts += hip_coords
            return torch.cat([hip_coords, kpts], dim=-2)
        else: 
            return self.rescale_to_hip_box(kpts)
    
    def transform_to_centered_visual_space(self, kpts):
        kpts = super().rescale_to_hip_box(kpts[...,len(self.node_hip):,:])
        return torch.cat([torch.zeros_like(kpts)[...,0,:].unsqueeze(-2), kpts],dim=-2)

    
class H36MSkeletonVelocity(H36MSkeleton, SkeletonVelocity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        
class H36MSkeletonUnitNorm(H36MSkeleton, SkeletonUnitNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


