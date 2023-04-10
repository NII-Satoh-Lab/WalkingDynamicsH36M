import torch
import numpy as np
from torch.utils import data

from .somof_dataset import SoMoFDataset
           
            
class SoMoFTorchDataset(data.Dataset):
    def __init__(self,
                 dataset: SoMoFDataset,
                 history_length: int,
                 prediction_horizon: int,
                 transform=lambda x: x,
                #  step: int = 1,
                #  skip_11_d: bool = False,
                #  action: str ="all",
                **kwargs
                 ):
        
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        assert history_length == 16
        assert prediction_horizon == 14
        self.transform = transform

        self._data = dataset
        print("Len of sample: ", self.__len__())
    


    def __getitem__(self, item):
        poses = self._data["poses_3d"][item]
        poses = self.transform(poses)
        return poses[:self.history_length], poses[self.history_length: self.history_length + self.prediction_horizon]

    def __len__(self):
        return len(self._data)

    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        THe poses are easy to mirrored because they are centered around 0 (the hip), so it`s just mirroring around x axis
        The hip motion needs to be mirrored only in direction of the hip
        """
        self._mirror(axis=0)
        # self._mirror(axis=2)
        
        
    def _mirror(self, axis=0):
        reversed_poses = self._data["poses_3d"].clone()
        reversed_poses[...,1:, axis] *= -1
        self._data["poses_3d"] = torch.cat([self._data["poses_3d"], reversed_poses], dim=-4)
        self._data["img_paths"] = self._data["img_paths"] + self._data["img_paths"]
        self._data["img_paths_in+out"] = self._data["img_paths_in+out"] + self._data["img_paths_in+out"]
        reversed_gt = self._data["gt"].clone()
        reversed_gt[...,1:, 0] *= -1
        self._data["gt"] = torch.cat([self._data["gt"], reversed_gt], dim=-4)
        
    def reverse(self):
        # Reverse sequence
        reverse_seq = self._data["poses_3d"].flip(-3)
        self._data["poses_3d"] = torch.cat([self._data["poses_3d"], reverse_seq], dim=-4)
        reverse_img_paths = np.flip(np.array(self._data["img_paths_in+out"]).copy(),-1)[:16].tolist()
        self._data["img_paths"] = self._data["img_paths"] + reverse_img_paths
        reverse_img_paths = np.flip(np.array(self._data["img_paths_in+out"]).copy(), -1).tolist()
        self._data["img_paths_in+out"] = self._data["img_paths_in+out"] + reverse_img_paths
        reversed_gt = self._data["gt"].flip(-3)
        self._data["gt"] = torch.cat([self._data["gt"], reversed_gt], dim=-4)

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }
        
        