import torch
import random

from .d3pw_dataset import D3PW_SoMoFDataset
from model.utils.dataset import DatasetAugmentation           
            
class D3PW_SoMoFTorchDataset(DatasetAugmentation):
    def __init__(self,
                 dataset: D3PW_SoMoFDataset,
                 history_length: int,
                 prediction_horizon_3dpw: int,
                 transform=lambda x: x,
                #  action: str ="all",
                **kwargs
                 ):
        
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon_3dpw
        assert history_length == 16
        assert self.prediction_horizon >= 14
        self.transform = transform

        self._data = dataset
        self.skeleton = self._data.skeleton

        print("Len of sample: ", self.__len__())
    


    def __getitem__(self, item):
        idx = item%len(self._data)
        poses = self._data["poses_3d"][idx]
        sample_len = self.history_length + self.prediction_horizon
        start_idx = random.randint(0, (len(poses)-sample_len*self._data.frame_dist))
        poses = torch.stack([poses[...,start_idx + i*self._data.frame_dist, :, :] 
                             for i in range(self.history_length+self.prediction_horizon)], dim=-3)
        
        poses = self.skeleton.transform_seq_w_hip_to_input_space(poses)
        poses = self.transform(poses)
        # idx_array = [start_idx+i*self._data.frame_dist for i in range(self.history_length+self.prediction_horizon) ]
        return poses[:self.history_length], poses[self.history_length: self.history_length + self.prediction_horizon]


    def __len__(self):
        return len(self._data)*self._data.multiply_time

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }
        
        