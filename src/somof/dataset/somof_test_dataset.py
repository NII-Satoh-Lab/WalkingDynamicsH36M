import torch
import numpy as np
import json
import os.path
from collections import UserDict

from somof.skeleton import SoMoFSkeleton
       
class SoMoFTestDataset(UserDict):
    """ """
    original_resolution = [1002, 1000]  # from video data
    images_subdir = 'ImageSequence'
    dataset_name = "3dpw" #"posetrack" 


    def __init__(self, dataset_path,  skeleton: SoMoFSkeleton, **kwargs):

        self.datase_path = dataset_path
        name_split = "test"
        dataset_name = "3dpw" #"posetrack"
        assert skeleton.num_joints == 13
        
        super().__init__(self._load(self.datase_path, dataset_name, skeleton, name_split))

        print(f'Successfully created SoMoF {dataset_name} dataset from file: ', dataset_path,
              '\n\tnumber of samples: ', len(self["poses_3d"]),
             )
    def __len__(self):
        return len(self["poses_3d"])
                
    @staticmethod
    def _load(dataset_path, dataset_name, skeleton, name_split="test"):
       
        with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_frames_in.json"), 'r') as f:
            frames_in = np.array(json.load(f)) #(221, 16)
            frames_in = [[os.path.join(dataset_path, "imageFiles", frame) for frame in frames]for frames in frames_in]
            
            # self.masks_in = np.array(json.load(f)) all ones
        with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_in.json"), 'r') as f:
            data_in = np.array(json.load(f)) #(85, 2, 16, 39)
        
        data = torch.from_numpy(data_in).view(data_in.shape[0]*data_in.shape[1], data_in.shape[2], 
                                                 skeleton.num_joints, 3).float()
        gt = torch.from_numpy(data_in)
        
        # from meters to mm
        data *= 1000
        data = skeleton.tranform_to_input_space_pose_only(data)
        hips = data[..., 0, :].unsqueeze(-2)
        pred_seq_start = hips[...,-1,:,:].unsqueeze(-3)
        kpts = skeleton.transform_seq_w_hip_to_input_space(data)
        result = {"poses_3d": kpts,
                  "seq_start": pred_seq_start, 
                  "img_paths": frames_in,
                  "input": gt}
        return result
        

