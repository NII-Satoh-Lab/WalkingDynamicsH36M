import torch
import math

from .h36m_dataset import H36MDataset, ACTIONS
from model.utils.dataset import DatasetAugmentation
           
class H36MTorchDataset(DatasetAugmentation):
    def __init__(self,
                 dataset: H36MDataset,
                 history_length: int,
                 prediction_horizon: int,
                 no_overlapping_samples=False, 
                 transform=lambda x: x,
                 action='all',
                 dataset_downsample_factor = 1,
                #  action: str ="all",
                **kwargs
                 ):
        self.no_overlapping_samples = no_overlapping_samples
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        
        assert dataset_downsample_factor > 0
        self.dataset_downsample_factor = dataset_downsample_factor
        self.fps = 50/dataset_downsample_factor
        self.transform = transform
        
        self.action = action
        assert action == "all" or action in ACTIONS

        self._data = dataset
    
        self.dataset_index = None
        self.determine_samples()

        print(f'Successfully created H36M dataloder',
              '\n\taction: ', action,
              '\n\tsubjects: ', list(self._data.subjects.keys()),
              '\n\tnumber of samples: ', self.__len__(),
              )
        
    def determine_samples(self):
        def extract_action(seq_name):
            return seq_name.split("/")[-1] #{subject}/{action}/{camera}
        span_of_single_sample = n_frames_spanned_by_sample(self.prediction_horizon, self.history_length, self.dataset_downsample_factor)
        if self.no_overlapping_samples:
            self.samples = [(s, frame*span_of_single_sample) for s, seq in enumerate(self._data["seq_names"]) 
                            if self.action in extract_action(seq) or self.action == "all"
                            for frame in range(math.floor(len(self._data["poses_3d"][s])/span_of_single_sample))]
        else:
            self.samples = [(s, frame) for s, seq in enumerate(self._data["seq_names"]) 
                            if self.action in extract_action(seq) or self.action == "all"
                            for frame in range(len(self._data["poses_3d"][s]) - span_of_single_sample + 1)]
        seq2idx = {}
        for idx, (seq, frame_idx) in enumerate(self.samples):
            if seq not in seq2idx:
                seq2idx[seq] = {frame_idx: idx}  
            else:
                seq2idx[seq][frame_idx] = idx
        self.seq2idx = seq2idx
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        seq, start_idx = self.samples[item]
        data = self._data["poses_3d"][seq]
        poses = torch.stack([data[...,start_idx + i*self.dataset_downsample_factor, :, :] 
                        for i in range(self.history_length+self.prediction_horizon)], dim=-3)  
        poses = self.transform(poses)
        poses = self._data.skeleton.transform_seq_w_hip_to_input_space(poses)
        return poses[:self.history_length], poses[self.history_length: self.history_length + self.prediction_horizon]

    def hparam(self) -> dict:
        return {
            'DATA_history_length': self.history_length,
            'DATA_prediction_horizon': self.prediction_horizon,
        }
        
        
def n_frames_spanned_by_sample(n_out_frames, n_in_frames, sampling_dist):
    return (n_in_frames + n_out_frames - 1) * sampling_dist + 1