
import torch
import numpy as np
import cdflib
import os.path
from typing import Literal
from collections import UserDict

from .joint_correspondence_utils import convert_h36m_joints_to25, convert_h36m_joints_to17
from wdh36m.skeleton import H36MSkeleton


train_subjects = {'S1': 1, 'S7': 7, 'S8': 8, 'S9': 9, 'S11': 11, }
val_subjects = {'S5': 5, 'S6': 6, }
camera_ids = ['54138969', '55011271', '58860488', '60457274']

ACTIONS = {  
        #    'Phoning',
        #      'Eating',
        #      'Smoking',
        #      'WalkDog',
             'WalkTogether',
             'Walking',}


class H36MDataset(UserDict):
    """ """
    original_resolution = [1002, 1000]  # from video data
    """
    folder structure 
    S{n}/
        ImageSequence
        MyPoseFeatures/
            D3_Positions
                Walking.cdf
                Walking_1.cdf 
                ....
            D3_Positions_mono/
                Walking.{camera_id}.cdf
    """
    images_subdir = 'ImageSequence'
    poses_subdir = 'MyPoseFeatures/D3_Positions'


    def __init__(self, dataset_path, skeleton: H36MSkeleton, action='all', 
                 num_joints=25, split="train", **kwargs):

        self.datase_file_path = dataset_path
        self.skeleton = skeleton
        assert split in ["train", "valid"]
        self.subjects = train_subjects if split=="train" else val_subjects
        assert action == "all" or action in ACTIONS, "This action does not exists in h36m"
        self.action = ACTIONS if action == "all" else action
        self.n_keypoints = num_joints
        assert self.n_keypoints in [32, 25, 17]  
             
        super().__init__(self._load(self.datase_file_path, skeleton, self.subjects, self.action if action == "all" else [action], n_kpts=self.n_keypoints))
        if not skeleton.if_consider_hip:
            self.n_keypoints -= 1
            
        print(f'Successfully created H36M dataset',
              '\n\tsplit: ', split,
              '\n\tnumber of sequences: ', len(self["poses_3d"]),
              )
        
        
    @staticmethod 
    def get_img_path_for_seq(seq, root_dir):
        subject = seq.split('/')[0]
        imgs_path = os.path.join(root_dir, subject, H36MDataset.images_subdir, seq.replace(f'{subject}/', ''), camera_ids[0])
        return imgs_path 
    
    @staticmethod 
    def select_keypoints_as_config(kpts, n_kpts):
        if n_kpts == 32:
            return kpts
        if n_kpts == 25:
            return convert_h36m_joints_to25(kpts)
        if n_kpts == 17:
            return convert_h36m_joints_to17(kpts)
        else: 
            assert 0
                
    @staticmethod
    def _load(dir_path, skeleton, subjects, actions, n_kpts=17):
        """returns a dict with img_paths, poses3d, hip3d, centered3d

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        datalist = []
        image_list = []
        seq_list = []
        for subj in subjects:
            for action in actions:
                path = os.path.join(dir_path, subj, H36MDataset.poses_subdir)
                action_names = [f for f in os.listdir(path) if f.endswith(".cdf") and action in f]
                for act in action_names:
                    with cdflib.CDF(os.path.join(path, act)) as cdf:
                        poses_3d_univ = np.array(cdf['Pose'])
                        poses_3d_univ = poses_3d_univ.reshape(poses_3d_univ.shape[1], 32, 3)
                    
                    kpts = torch.from_numpy(H36MDataset.select_keypoints_as_config(poses_3d_univ, n_kpts=n_kpts)).float()
                    kpts = torch.stack([kpts[..., 0], kpts[..., 2], kpts[..., 1]], dim=-1)
                    kpts = skeleton.tranform_to_input_space_pose_only(kpts)
                    seq_name = os.path.join(subj, act.strip(".cdf"))
                    rgb_folder_path = H36MDataset.get_img_path_for_seq(seq_name, path)
                    # subject, action, camera = seq_name.split("/")
                    img_paths = [os.path.join(rgb_folder_path, 'img_%06d.jpg' % (frame_idx + 1)) for frame_idx in
                                    range(kpts.shape[0])]
                    datalist.append(kpts)
                    image_list.append(img_paths)
                    seq_list.append(seq_name)
        result = {  "poses_3d": datalist,
                    "img_paths": image_list,
                    "seq_names": seq_list}
        return result
        
 
        
    def __getitem__(self, item):
        return  self.data[item]  
        
        
