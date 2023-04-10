import torch
import numpy as np
import json
import os.path
from collections import UserDict

from somof.skeleton import SoMoFSkeleton
       
class SoMoFDataset(UserDict):
    """ """
    original_resolution = [1002, 1000]  # from video data
    images_subdir = 'ImageSequence'
    dataset_name = "3dpw" #"posetrack" 


    def __init__(self, dataset_path,  skeleton: SoMoFSkeleton,
                 num_joints=13, split="train", **kwargs):

        self.datase_path = dataset_path
        assert split in ["train", "valid"]
        name_split = split
        dataset_name = "3dpw" #"posetrack"
        assert num_joints == 13
        
        self.skeleton = skeleton
        # self.n_keypoints = 13
        
        super().__init__(self._load(self.datase_path, dataset_name, skeleton, name_split))

        print(f'Successfully created SoMoF {dataset_name} dataset from file: ', dataset_path,
              '\n\tsplit: ', split,
              '\n\tnumber of samples: ', len(self["poses_3d"]),
            )
    def __len__(self):
        return len(self["poses_3d"])
                
    @staticmethod
    def _load(dataset_path, dataset_name, skeleton, name_split="train"):
        
        with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_frames_in.json"), 'r') as f:
            frames_in = np.array(json.load(f)) #(221, 16)
            frames_in = [[os.path.join(dataset_path, "imageFiles", frame) for frame in frames]for frames in frames_in]
            
        last_in_frame_idx = [int(frames[-1].split("/")[-1].replace(".jpg", "").replace("image_", "")) for frames in frames_in]
        frames_out = [["/".join(frames_in[i][0].split("/")[:-1])+f"/image_{idx + f*2:05d}.jpg" for f in range(1,15)] for i,idx in enumerate(last_in_frame_idx)]
        frames = [inf+ out for inf,out in zip(frames_in, frames_out)]
            # self.masks_in = np.array(json.load(f)) all ones
        with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_in.json"), 'r') as f:
            data_in = np.array(json.load(f)) #(221, 2, 16, 39)
        # with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_masks_out.json"), 'r') as f:
        #     self.masks_out = json.load(f)
        with open(os.path.join(dataset_path, f"{dataset_name}_{name_split}_out.json"), 'r') as f:
            data_out = np.array(json.load(f)) #(221, 2, 14, 39)
            
        correspondence = np.array([[i]*len(data_in[i]) for i in range(len(frames_in))]).flatten()
        data_in = torch.from_numpy(data_in).view(data_in.shape[0]*data_in.shape[1], data_in.shape[2], 
                                                 skeleton.num_joints, 3).float()
        data_out = torch.from_numpy(data_out).view(data_out.shape[0]*data_out.shape[1], data_out.shape[2], 
                                                 skeleton.num_joints, 3).float()
        
        data = torch.cat([data_in, data_out], dim=-3)
        # from meters to mm
        data *= 1000
        kpts = skeleton.tranform_to_input_space(data)
        result = {"poses_3d": kpts,
                  "img_paths": frames_in,
                  "img_paths_in+out": frames,
                  "correspondence" : correspondence,
                  "gt": torch.cat([data_in, data_out], dim=-3)}
        return result
        

