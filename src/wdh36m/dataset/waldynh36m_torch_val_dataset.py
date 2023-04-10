import torch

from .h36m_dataset import H36MDataset
           
class WalDynH36MTorchValDataset():
    def __init__(self,
                 dataset: H36MDataset,
                 history_length: int,
                 prediction_horizon: int,
                 action='all',
                #  action: str ="all",
                **kwargs
                 ):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        
        self.action = "Custom"
        # assert action == "all" or action in ACTIONS

        self._data = dataset
        print(f'Successfully created H36M dataloder',
              '\n\taction: ', action,
              '\n\tnumber of samples: ', self.__len__(),
              )
        
    def __len__(self):
        return len(self._data["poses_3d"])
        

    def __getitem__(self, item):
        data = self._data["poses_3d"][item]
        poses = self._data.skeleton.transform_seq_w_hip_to_input_space(data)
        return poses[:self.history_length], poses[self.history_length: self.history_length + self.prediction_horizon]

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }
        
