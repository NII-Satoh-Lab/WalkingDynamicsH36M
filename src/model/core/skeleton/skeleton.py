from typing import List

import numpy as np
import torch
import random

from model.utils.keypoints import center_kpts_around_hip, random_rotation_y_axis

    
class Skeleton():
    node_hip = {...}  # fill up

    def __init__(self, if_consider_hip=True, pose_box_size=1100, history_length=50, **kwargs):
        
        self.if_consider_hip = if_consider_hip
        self.pose_box_size = pose_box_size
        self.history_length = history_length
        # TO DO define following variables
        # self.joint_dict_orig  = {}
        # self.limbseq = [[0, 1], ...
        #                 ]

        # if if_consider_hip:
        #     self.node_dict = {**Skeleton.node_hip, **{i +len(Skeleton.node_hip): v for i,v in self.joint_dict_orig.items()}}
        # else: 
        #     self.node_dict =  self.joint_dict_orig
        # self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]

    @property
    def num_joints(self):
        return len(self.joint_dict_orig)

    @property
    def num_nodes(self):
        return len(self.node_dict)
    
    @property
    def left_right_limb(self):
        return self.left_right_limb_list.copy()
    
    @property
    def nodes_type_id(self):
        joint_id_string_wo = []
        for joint_id_string in list(self.node_dict.values()):
            if joint_id_string[0] == 'L' and joint_id_string[1].isupper():
                joint_id_string_wo.append(joint_id_string[1:])
            elif joint_id_string[0] == 'R' and joint_id_string[1].isupper():
                joint_id_string_wo.append(joint_id_string[1:])
            else:
                joint_id_string_wo.append(joint_id_string)
        unique_strings = list(dict.fromkeys(joint_id_string_wo))
        joint_ids = [unique_strings.index(s) for s in joint_id_string_wo]
        return torch.tensor(joint_ids)
    
    def extract_limb_length(self, kpts):
        limbdist = []
        for l1,l2 in self.limbseq:
            limbdist.append( (kpts[..., l1, :] - kpts[..., l2, :]).norm(dim=-1))
        return torch.stack(limbdist, dim=-1)

    #################  functions to obtain input poses ########################################################################################
    
    def transform_seq_w_hip_to_input_space(self, data):
        kpts =  ...
        return kpts
    
    def tranform_to_input_space(self, data):
        data = self.tranform_to_input_space_pose_only(data)
        # centered, hips = data[..., 1:, :], data[..., 0, :].unsqueeze(-2)
        if not self.if_consider_hip:
            kpts = data[..., 1:, :]
        else: 
            kpts = self.transform_seq_w_hip_to_input_space(data)
        return kpts
    
    def tranform_to_input_space_pose_only(self, data):
        #center kpts
        centered, hips = center_kpts_around_hip(data, hip_idx=0)
        
        lower_box_shift = torch.ones((3))*self.pose_box_size # in mm
        centered /= lower_box_shift
        assert (centered >= -1.).all() and (centered <= 1.).all(), f"{centered.max()} {centered.min()}"

        kpts = torch.cat([hips, centered[...,1:,:]], dim=-2)
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
    
    def transform_hip_to_metric_space(self, kpts):
        hip_coords = ...
        ... 
        return hip_coords

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
            return kpts
        else: 
            return self.rescale_to_hip_box(kpts)
    
    def transform_to_test_space(self, kpts, seq_start):
        """_summary_
        """
        kpts = self.transform_to_metric_space(kpts)
        assert self.if_consider_hip
        kpts += seq_start
        kpts /= 1000 # from mm to m
        return kpts
        
    def rescale_to_hip_box(self, kpts):
        kpts = kpts.clone()
        
        lower_box_shift = self.pose_box_size # in mm
        # range [-1,1] around hips
        kpts *= lower_box_shift # scaled back to box around kpts
        return kpts
    
    ##########################  function to tranform back to visual space ##########################################################################

    def transform_to_centered_visual_space(self, kpts):
        kpts = self.rescale_to_hip_box(kpts[...,len(self.node_hip):,:])
        return kpts
    
    def transform_to_metric_space_for_visual(self, kpts):
        kpts = self.transform_to_metric_space(kpts)
        return kpts
    
    ##########################  data augmentation function ##########################################################################

    def rescale_body(self, kpts):
        ...
        return kpts
    
    def rotate_around_y_axis(self, kpts):
        # do this before splitting hips in norm & unit vector, if it is pose residual or not it does not matter
        return random_rotation_y_axis(kpts)
        

