import torch
import numpy as np
from functools import partial

from .plot import get_np_frames_3d_projection
from .image import load_image
from ..core.skeleton.skeleton_velocity import SkeletonVelocity

def get_color_frames_in_out(track, n_in_frames, **kwargs):
    frames_in = get_np_frames_3d_projection(track[..., :n_in_frames, :, :],**kwargs)
    fake_start = torch.from_numpy(track[0]).unsqueeze(0).broadcast_to(track.shape).numpy()
    if "data_pred" in kwargs:
        kwargs.pop("data_pred")
    frames_pred = get_np_frames_3d_projection(fake_start,data_pred=track, **kwargs)
    frames = frames_in + frames_pred[n_in_frames:]
    return frames

def visuals_step_3D_hc(skeleton, input, kpts, gt, imgs_path=None):
    
    torch.set_printoptions(sci_mode=False)
    
    n_in_frames, n_out_frames = input.shape[-3], kpts.shape[-3]
    get_color_frames_in_out_f = partial(get_color_frames_in_out, n_in_frames=n_in_frames)
    #50,24,3 25,24,3
    limbseq = skeleton.limbseq
    left_right_limb = skeleton.left_right_limb
    if not skeleton.if_consider_hip:
        input = skeleton.transform_to_centered_visual_space(input)
        gt = skeleton.transform_to_centered_visual_space(gt)
        kpts = skeleton.transform_to_centered_visual_space(kpts)
    else:
        input_hip = skeleton.transform_to_metric_space_for_visual(input)  
        kpts_hip = skeleton.transform_to_metric_space_for_visual(kpts)
        gt_hip = skeleton.transform_to_metric_space_for_visual(gt)
        if  isinstance(skeleton, SkeletonVelocity):
            kpts_hip += input_hip[...,-1,0,:]
            gt_hip += input_hip[...,-1,0,:]        

        kpts = skeleton.transform_to_centered_visual_space(kpts)
        gt = skeleton.transform_to_centered_visual_space(gt)
        input = skeleton.transform_to_centered_visual_space(input)        
        
    # puoi anche sovrapporre
    kpts_in = torch.cat([input, kpts], dim=0)
    gt_in = torch.cat([input, gt], dim=0)
    if skeleton.if_consider_hip:
        kpts_hip_in = torch.cat([input_hip, kpts_hip], dim=0)
        gt_hip_in = torch.cat([input_hip, gt_hip], dim=0)
        kpts_hip_in = kpts_hip_in - kpts_hip_in[..., 0,0,:]
        gt_hip_in = gt_hip_in - gt_hip_in[..., 0,0,:]
        
        
    images = {}

    def show_in_horizontal_columns(tensor_list):
        return torch.cat(tensor_list, dim=-1)
    def show_in_vertical_rows(tensor_list):
        return torch.cat(tensor_list, dim=-2)
    
    dataset = "h36m"
    
    if imgs_path is not None:
        rgb_seq = np.array([load_image(img_path) for img_path in imgs_path])
        images["RGB seq input gif"] = rgb_seq
    
        n_in_frames, n_out_frames = input.shape[-3], kpts.shape[-3]

 
    
    out_kpts_frames = get_color_frames_in_out_f(kpts.numpy(),limbseq=limbseq, left_right_limb=left_right_limb, 
                                                  xyz_range=None, center_pose=False, units="mm", 
                                                as_tensor=True, orientation_like=dataset, title="Output Kpts")
    target_kpts_frames  = get_np_frames_3d_projection(gt.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                      xyz_range=None, center_pose=False, units="mm", 
                                                as_tensor=True, orientation_like=dataset, title="Target Kpts")
    in_kpts_frames = get_color_frames_in_out_f(kpts_in.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                           data_pred=None, xyz_range=None, 
                                                           center_pose=False, units="mm", as_tensor=True, 
                                                           orientation_like=dataset, title="Out&In Kpts")
    in_gt_frames = get_np_frames_3d_projection(gt_in.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                        data_pred=None, xyz_range=None, 
                                                        center_pose=False, units="mm", as_tensor=True, 
                                                        orientation_like=dataset, title="Target&in Kpts")
    if skeleton.if_consider_hip:
        out_kpts_cam_frames = get_color_frames_in_out_f(kpts_hip.numpy(),limbseq=limbseq, left_right_limb=left_right_limb, 
                                                    xyz_range=None, center_pose=False, units="mm", 
                                                    as_tensor=True, orientation_like=dataset, title=f"Output Kpts WITH HIP\nCenter: {kpts_hip[0,0].numpy()/1000}")
        out_kpts_cam_frames_same_orientation = get_color_frames_in_out_f(kpts_hip.numpy(),limbseq=limbseq, left_right_limb=left_right_limb, 
                                                    xyz_range=None, center_pose=False, units="mm", 
                                                    as_tensor=True, orientation_like=dataset, title=f"Output Kpts WITH HIP gt orientation\nCenter: {kpts_hip[0,0].numpy()/1000}",
                                                    center_like=gt_hip.numpy())
        target_kpts_cam_frames  = get_np_frames_3d_projection(gt_hip.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                        xyz_range=None, center_pose=False, units="mm", 
                                                    as_tensor=True, orientation_like=dataset, title="Target Kpts WITH HIP")
        in_kpts_cam_frames = get_color_frames_in_out_f(kpts_hip_in.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                            data_pred=None, xyz_range=None, 
                                                            center_pose=False, units="mm", as_tensor=True, 
                                                            orientation_like=dataset, title="Out&In Kpts  WITH HIP")
        in_gt_cam_frames = get_np_frames_3d_projection(gt_hip_in.numpy(), limbseq=limbseq, left_right_limb=left_right_limb, 
                                                            data_pred=None, xyz_range=None, 
                                                            center_pose=False, units="mm", as_tensor=True, 
                                                            orientation_like=dataset, title="Target&in Kpts  WITH HIP")


    def create_gif_tensor(list_of_tensors):
        return show_in_horizontal_columns([torch.stack(t,dim=0) for t in list_of_tensors])
    
    def create_img_tensor(list_of_tensors):
        return show_in_vertical_rows([show_in_horizontal_columns(t) for t in list_of_tensors])
    
        
    # images["Output and GT"] = create_img_tensor(tensor_list)
    # images["Overlapping"] = create_img_tensor([out_kpts_frames, overlappping_kpts_frames])
    images["Prediction and GT - centered gif"] = create_gif_tensor([out_kpts_frames, target_kpts_frames])
    images["Prediction and GT with past - centered gif"] = create_gif_tensor([in_kpts_frames, in_gt_frames])
    if skeleton.if_consider_hip:
        images["Prediction and GT - with global translation gif"] = create_gif_tensor([out_kpts_cam_frames, out_kpts_cam_frames_same_orientation, target_kpts_cam_frames])
        images["Prediction and GT with past - with global translation gif"] = create_gif_tensor([in_kpts_cam_frames, in_gt_cam_frames])
    

    return images