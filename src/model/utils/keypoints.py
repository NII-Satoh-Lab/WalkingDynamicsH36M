def center_kpts_around_hip(kpts, hip_idx=0):
    #hip is always joint n0 in h36m regardless of number of joints considered
    center = kpts[..., hip_idx, :].unsqueeze(-2) #(n_frames, n_joints, 3)
    centered_kpts = kpts - center
    # centered_kpts = centered_kpts[..., 1:, :] # keep hip joint for compatibility with rest of code
    return centered_kpts, center

import random
import torch
import math

def random_rotation_y_axis(kpts, axis=1):
    angle = random.random()*360
    phi = torch.tensor(angle * math.pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.tensor([[c, 0, -s],
                        [0, 1.0, 0],
                        [s, 0, c]]) # 3x3

    x_rot = kpts @ rot.t() 
    return x_rot
