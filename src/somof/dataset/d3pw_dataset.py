import torch
import numpy as np
import os.path
from collections import UserDict
import pickle

from somof.skeleton import SoMoFSkeleton
       
class D3PW_SoMoFDataset(UserDict):
    """ """
    # TRAIN AND TEST SETS ARE REVERSED FOR SOMOF
    SPLIT_3DPW = {
        "train": "test",
        "val": "validation",
        "valid": "validation",
        "test": "train"
    }
    
    @staticmethod
    def convert_3dpw2somof(kpts):
        """        
        JOINTS_DICT_SMPL_24  = {
                0: 'Pelvis', 1: "LHip", 2: "RHip", 
                3: 'Waist', 6: 'Torax', 9: 'Chest', 
                4: 'LKnee', 5: 'RKnee',
                7: 'LAnkle', 8: 'RAnkle',
                10: 'LToes', 11: 'RToes', 
                12: "Neck", 
                13: "LClavicle", 14: "RClavicle",
                15: 'Nose',
                16: "LShoulder", 17: "RShoulder", 18: "LElbow", 19: "RElbow",
                20: "LWrist", 22: "LHand",
                21: "RWrist", 23: "RHand",
                }
                
        joint_dict_somof  = {0: 'LHip', 1: 'RHip',
                     2: 'LKnee',  3: 'RKnee', 4: 'LAnkle', 5: 'RAnkle',
                     6: 'Head',
                     7: 'LShoulder', 8: 'RShoulder',
                     9: 'LElbow', 10: 'RElbow', 11: 'LWrist', 12: 'RWrist'}
        """
        CONVERSION_IDX_SMPL24_TO_SOMOF = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
        kpts = torch.stack([kpts[..., i, :] for i in CONVERSION_IDX_SMPL24_TO_SOMOF], dim=-2)
        assert kpts.shape[-2] == 13
        return kpts


    def __init__(self, dataset_path_3dpw, skeleton: SoMoFSkeleton, split="train", **kwargs):
        

        image_path = os.path.join(dataset_path_3dpw, 'imageFiles')
        seq_path = os.path.join(dataset_path_3dpw, 'sequenceFiles', D3PW_SoMoFDataset.SPLIT_3DPW[split])
        assert skeleton.num_joints == 13
        self.skeleton = skeleton
        
        self.multiply_time = 12
        self.frame_dist  = 2
        # self.n_keypoints = 13
        
        super().__init__(self._load(seq_path, image_path, skeleton=skeleton))

        print(f'Successfully created 3DPW with SoMoF settings',
              '\n\tsplit: ', split,
              '\n\tnumber of samples: ', len(self["poses_3d"]),
            )
    def __len__(self):
        return len(self["poses_3d"])
                
    @staticmethod
    def _load(dataset_path, image_path, skeleton):
        datalist = []
        image_list = []
        for pkl in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, pkl), 'rb') as reader:
                annotations = pickle.load(reader, encoding='latin1')
            # all_person_tracks = []
            # imgs_track = []
            genders = annotations['genders']
            for actor_index in range(len(annotations['genders'])):

                # joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
                joints_3D = np.array(annotations['jointPositions'][actor_index])
                joints_3D = joints_3D.reshape(joints_3D.shape[0], -1, 3)
                # track_mask = []

                imgs = [os.path.join(image_path, os.path.splitext(pkl)[0], f"image_{str(img_idx).zfill(5)}.jpg") 
                        for img_idx in range(len(joints_3D))]
                # J_3D_mask = np.ones(J_3D_real.shape[:-1])
                # track_mask.append(J_3D_mask)

                # all_person_tracks.append(np.asarray(J_3D_real))
                # imgs_track.append(imgs)
                joints_3D = torch.from_numpy(joints_3D).float()
                joints_3D *= 1000 # from meters to mm
                joints_3D = D3PW_SoMoFDataset.convert_3dpw2somof(joints_3D)
                kpts = skeleton.tranform_to_input_space_pose_only(joints_3D) #, consider_hip

                datalist.append(kpts)
                image_list.append(imgs)
        

        result = {"poses_3d": datalist,
                  "img_paths": image_list}
        return result
