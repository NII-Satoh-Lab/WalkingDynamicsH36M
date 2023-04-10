import os
import numpy as np
import json
import torch
import cdflib

import sys
sys.path.append(os.path.abspath('../src'))
from wdh36m.dataset.joint_correspondence_utils import convert_h36m_joints_to25


def track_name_to_h36m_cdf_path(name):
    #'S5/Smoking' --> "S5/MyPoseFeatures/D3_Positions/Smoking.cdf"
    return name.replace("/", "/MyPoseFeatures/D3_Positions/")+".cdf"


def mirror(kpts, axis=0):
    kpts = kpts.clone()
    kpts[..., axis] *= -1
    return kpts

def load_intrinsics(dataset_path, split):
    with open(os.path.join(dataset_path, f"{split}_frames_idxs.json"), 'r') as f:
        frame_idxs = np.array(json.load(f)) #(221, 2, 16, 39)
    with open(os.path.join(dataset_path, f"{split}_intrinsics.json"), 'r') as f:
        track_intrinsics = json.load(f) #(221, 2, 16, 39)
    with open(os.path.join(dataset_path, f"{split}_images.json"), 'r') as f:
        _imgs_ = json.load(f) #(221, 16)
        track_names = [fs[0].split("img")[0][:-1] for fs in _imgs_] # 'S5/Smoking'
    return frame_idxs, track_intrinsics, track_names
    
def load_poses(h36m_data_path, frame_idxs, track_intrinsics, track_names):
    wdh36m_poses = []

    for idx, track in enumerate(track_names):
        path = os.path.join(h36m_data_path, track_name_to_h36m_cdf_path(track))
        with cdflib.CDF(path) as cdf:
            poses_3d_univ = np.array(cdf['Pose'])
            poses_3d_univ = poses_3d_univ.reshape(poses_3d_univ.shape[1], 32, 3)
        
        kpts = torch.from_numpy(convert_h36m_joints_to25(poses_3d_univ)).float()
        kpts = torch.stack([kpts[..., 0], kpts[..., 2], kpts[..., 1]], dim=-1)
        kpts = kpts[frame_idxs[idx]]
        if track_intrinsics[idx] != "None":
            kpts = mirror(kpts, axis=0 if track_intrinsics[idx] == "x" else 2)
        wdh36m_poses.append(kpts)
    return torch.stack(wdh36m_poses, dim=0)


def create_wdh36m_tracks(wdh36m_dataset_path, h36m_data_path):
    for split in ["val", "test"]:
        intrinsics = load_intrinsics(wdh36m_dataset_path, split)
        wdh36m_poses = load_poses(h36m_data_path, *intrinsics)
        
        with open(os.path.join(wdh36m_dataset_path, f"{split}_poses.json"), 'w') as f:
            f.write(json.dumps(wdh36m_poses.numpy().tolist()))

    

import argparse
parser = argparse.ArgumentParser(description='Extract H3.6M poses from WalkingDynamicsH36M intrinsics')
parser.add_argument('--h36m_path',
                    type=str,
                    help='Path to original H36M dataset',
                    default='../data/h36m')

parser.add_argument('--wdh36m_path',
                    type=str,
                    help='Path to WalkingDynamicsH36M dataset intrisics files',
                    default='./')

args = parser.parse_args()

   
if __name__ == '__main__':
    create_wdh36m_tracks(args.wdh36m_path, args.h36m_path)
    print("Tracks generated successfully")