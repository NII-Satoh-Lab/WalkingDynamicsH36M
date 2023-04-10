import torch
from torch.utils import data
import random

from .h36m_dataset import H36MDataset, ACTIONS

class H36MTorchValDataset(data.Dataset):
    def __init__(self,
                 dataset: H36MDataset,
                 history_length: int,
                 prediction_horizon: int,
                 transform=lambda x: x,
                 action='all',
                 dataset_downsample_factor = 1,
                 num_samples: int = 256,
                #  action: str ="all",
                **kwargs
                 ):
        self.no_overlapping_samples = True
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        
        assert dataset_downsample_factor > 0
        self.dataset_dataset_downsample_factor = dataset_downsample_factor
        self.fps = 50/dataset_downsample_factor
        self.transform = transform
        
        self.action = action
        assert action == "all" or action in ACTIONS

        self._data = dataset
    
        self.dataset_index = None
        self.num_samples = num_samples
        self.determine_samples()
        print(f'Successfully created H36M validation dataloder',
              '\n\taction: ', action,
              '\n\tsubjects: ', list(self._data.subjects.keys()),
              '\n\tnumber of samples: ', self.__len__(),
              )

    def __len__(self):
        return len(self.data_poses)
        
    def determine_samples(self):
        if self.action == "all":
            self.dataset_index = [i for i in range(len(self._data["seq_names"]))]
        else:
            self.dataset_index = []
            for s, sub_action in enumerate(self._data["seq_names"]):
                if self.action in sub_action:
                    self.dataset_index.append(s)
            assert len(self.dataset_index) > 0, len(self.dataset_index)
        sample_len = self.history_length + self.prediction_horizon
        self.data_poses = []
        n = 0
        self.seq_start_idx = []
        while n < self.num_samples:
            i = n%len(self.dataset_index)
            poses = self._data["poses_3d"][self.dataset_index[i]]
            start_idx = random.randint(0, 
                                            (len(poses)-sample_len*self.dataset_dataset_downsample_factor))    
            poses = torch.stack([poses[...,start_idx + i*self.dataset_dataset_downsample_factor, :, :] 
                             for i in range(self.history_length+self.prediction_horizon)], dim=-3)
    
            poses = self._data.skeleton.transform_seq_w_hip_to_input_space(poses)
            poses = self.transform(poses)
            self.data_poses.append(poses)
            self.seq_start_idx.append((self.dataset_index[i], start_idx))
            n += 1

    def __getitem__(self, item):
        # seq = self._data["seq_names"][self.seq_start_idx[item][0]]
        poses = self.data_poses[item]
        return poses[:self.history_length], poses[self.history_length: self.history_length + self.prediction_horizon]

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }