import torch
from torch.utils import data

from .somof_test_dataset import SoMoFTestDataset
           
            
class SoMoFTorchTestDataset(data.Dataset):
    def __init__(self,
                 dataset: SoMoFTestDataset,
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
    


    def __getitem__(self, item):
        poses = self._data["poses_3d"][item]
        seq_start = self._data["seq_start"][item]
        return poses, seq_start

    def __len__(self):
        return len(self._data)

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }
        
        