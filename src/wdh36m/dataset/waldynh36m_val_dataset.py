import torch
import numpy as np
import json
import os.path
from collections import UserDict

from wdh36m.skeleton import H36MSkeleton

class WalDynH36MValDataset(UserDict):
    """ """

    def __init__(self, dataset_val_path, skeleton: H36MSkeleton,
                 num_joints=25, **kwargs):

        self.datase_file_path = dataset_val_path
        self.skeleton = skeleton
        self.subjects = ...
        # assert action == "all" or action in ACTIONS, "This action does not exists in h36m"
        # self.action = ACTIONS if action == "all" else action
        self.n_keypoints = num_joints
        assert self.n_keypoints in [32, 25, 17]  

        
        self.skeleton = skeleton
        split = kwargs["split"]
        
        super().__init__(self._load(self.datase_file_path, skeleton, num_joints, split))

        print(f'Successfully created H36M dataset',
              '\n\tsplit: val my dataset',
              '\n\tnumber of sequences: ', len(self["poses_3d"]),
              )
        
    def __len__(self):
        return len(self["poses_3d"])
                
    @staticmethod
    def _load(dataset_path, skeleton, n_kpts, split):
       
        with open(os.path.join(dataset_path, f"{split}_poses.json"), 'r') as f:
            data = np.array(json.load(f)) #(221, 2, 16, 39)
        with open(os.path.join(dataset_path, f"{split}_images.json"), 'r') as f:
            frames = np.array(json.load(f)) #(221, 16)
        seqs = [fs[0].split("img")[0][:-1] for fs in frames]
        assert n_kpts == data.shape[-2]
        kpts = torch.from_numpy(data).float()
        kpts = skeleton.tranform_to_input_space_pose_only(kpts)
        result = {  "poses_3d": kpts,
            "img_paths": frames,
            "seq_names": seqs}
        return result
        

