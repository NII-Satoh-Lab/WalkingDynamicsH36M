import torch
from torch.utils import data
import numpy as np

class DatasetAugmentation(data.Dataset):
    
    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        THe poses are easy to mirrored because they are centered around 0 (the hip), so it`s just mirroring around x axis
        The hip motion needs to be mirrored only in direction of the hip
        """
        self._mirror(axis=0)
        # self._mirror(axis=2)
        
        
    def _mirror(self, axis=0):
        reversed_poses = [] #[poses.clone() for poses in self._data["poses_3d"]]
        for poses in self._data["poses_3d"]:
            poses = poses.clone()
            poses[..., axis] *= -1
            reversed_poses.append(poses)
        self._data["poses_3d"] = self._data["poses_3d"] + reversed_poses 
        self._data["img_paths"] = self._data["img_paths"] + self._data["img_paths"]
        
    def reverse(self):
        # Reverse sequence
        reverse_seq = [] #[poses.clone() for poses in self._data["poses_3d"]]
        for poses in self._data["poses_3d"]:
            poses = poses.flip(-3)
            reverse_seq.append(poses)
        self._data["poses_3d"] = self._data["poses_3d"] + reverse_seq
        reverse_img_paths = np.flip(np.array(self._data["img_paths"], dtype=object).copy(),-1).tolist()
        self._data["img_paths"] = self._data["img_paths"] + reverse_img_paths
        