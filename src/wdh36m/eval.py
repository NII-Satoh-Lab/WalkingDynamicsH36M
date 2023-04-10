import os
import yaml
from typing import Sequence
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, EpochOutputStore
from ignite.engine import Events, Engine, DeterministicEngine
from ignite.metrics import Loss

import sys
sys.path.append(os.path.abspath('./'))

from skeleton import H36MSkeletonUnitNorm, H36MSkeletonVelocity
from dataset.h36m_dataset import H36MDataset
from dataset.waldynh36m_val_dataset import WalDynH36MValDataset
from wdh36m.dataset.waldynh36m_torch_val_dataset import WalDynH36MTorchValDataset
from model.core.metrics.difference import Difference
from model.utils.torch import set_seed
from model.network import Motion
from model.core.metrics.mpjpe import MeanPerJointPositionError
from model.core.metrics.fde import FinalDisplacementError
from model.core.metrics.vim import VIM
from model.utils.visuals import visuals_step_3D_hc
from model.utils.image import save_img, save_gif


def eval(output_log_path: str,
          batch_size_eval: int = 0,
          device: str = 'cpu',
          seed: int = 0,
          detect_anomaly: bool = False,
          **kwargs) -> None:
    """
    Evaluate
    """
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)

    # Load skeleton configuration
    if kwargs["hip_mode"] == "norm":
        skeleton = H36MSkeletonUnitNorm(**kwargs)
    elif kwargs["hip_mode"] == "velocity": 
        skeleton = H36MSkeletonVelocity(**kwargs)
    else: 
        assert 0, "Mode non iplemented"    

    # Create model
    model = Motion(skeleton.num_nodes,
                   T=skeleton.nodes_type_id, **kwargs).to(device)

    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Load the H3.6M Dataset from disk
    # h36m_dataset_train = H36MDataset(split="train", skeleton=skeleton, **kwargs)
    h36m_dataset_eval = WalDynH36MValDataset(skeleton=skeleton, **kwargs)

    prediction_horizon_train = kwargs['prediction_horizon_train']
    prediction_horizon_eval = kwargs['prediction_horizon_eval']
    history_length = kwargs['history_length']

    # Create TorchDatasets for evalution and validation    
    dataset_eval = WalDynH36MTorchValDataset(h36m_dataset_eval, **kwargs)
    data_loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=batch_size_eval, num_workers=0)

    checkpoint = torch.load(kwargs['load_path'])
    model.load_state_dict(checkpoint, strict=True)
    
    # Define pre-process function to transform input to correct data type
    def preprocess(engine: Engine):
        engine.state.batch =  [t.to(device) for t in engine.state.batch[:]]
        
    def validation_step_eval(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            x, y = batch
            model_out, _, _ = model(x, prediction_horizon_eval, None)
            return model_out, y, x
        
    def rescale_output(output):
        output = [skeleton.transform_to_metric_space(out) for out in output[:2]]
        return output[0], output[1]
    
    def extract_limb_length(output):
        output = [skeleton.extract_limb_length(skeleton.transform_to_metric_space(out)) for out in output[:2]]
        
        return output[0], output[1]
    def rescale_output_no_hip(output):
        output = [skeleton.rescale_to_hip_box(out[...,len(skeleton.node_hip):,:]) for out in output[:2]]
        return output[0], output[1]
    
    def rescale_output_hip_only(output):
        output = [skeleton.transform_to_metric_space(out)[..., 0:1, :] for out in output[:2]]
        return output[0], output[1]
    
    # Define ignite metrics
    mpjpe = MeanPerJointPositionError(output_transform=rescale_output)
    mpjpe_pose = MeanPerJointPositionError(output_transform=rescale_output_no_hip)
    mpjpe_avg = MeanPerJointPositionError(output_transform=rescale_output, keep_time_dim=False)
    mpjpe_hip = MeanPerJointPositionError(output_transform=rescale_output_hip_only, keep_time_dim=True)
    limbs_dist = Difference(output_transform=extract_limb_length, keep_time_dim=True, keep_joint_dim=True)

    fde = FinalDisplacementError(output_transform=rescale_output)
    fde_pose = FinalDisplacementError(output_transform=rescale_output_no_hip)
    fde_hip = FinalDisplacementError(output_transform=rescale_output_hip_only)

    vim = VIM(output_transform=rescale_output_no_hip, dataset_name="3dpw")


    loss_metric_pose = Loss(model.loss_pose, output_transform=lambda x: (x[0], x[1]))
    loss_metric_hip = Loss(model.loss_hip, output_transform=lambda x: (x[0], x[1]))


    evaluator = DeterministicEngine(validation_step_eval)
    evaluator.add_event_handler(Events.ITERATION_STARTED, preprocess)
    mpjpe.attach(evaluator, 'MPJPE')
    mpjpe_pose.attach(evaluator, 'MPJPE_POSE')
    mpjpe_avg.attach(evaluator, 'MPJPE_AVG')
    mpjpe_hip.attach(evaluator, 'MPJPE_HIP')
    loss_metric_pose.attach(evaluator, 'Loss Pose')
    loss_metric_hip.attach(evaluator, 'Loss Hip')
    fde.attach(evaluator, 'FDE')
    fde_pose.attach(evaluator, 'FDE_POSE')
    fde_hip.attach(evaluator, 'FDE_HIP')
    vim.attach(evaluator, 'VIM')
    limbs_dist.attach(evaluator, 'LIMB dist GT')

    
    eos = EpochOutputStore()
    eos.attach(evaluator, 'output')

    pbar = ProgressBar()
    pbar.attach(evaluator)
    
    def save_visuals(model_out, y, x, idx, save_path, mode="train", imgs_path=None):
        image_dict = visuals_step_3D_hc(skeleton, x[idx].cpu(), model_out[idx].cpu(), y[idx].cpu(), imgs_path=imgs_path)
        for name, image in image_dict.items():
            fpath = os.path.join(save_path, f"img_{name}")
            save_img(image, fpath) if "gif" not in name else save_gif(image, name=fpath, fps=5)
            
    def print_trajectories(out, y, x, save_path):
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        n_subplots = int(len(y)**0.5) + 1
        n_tracks_per_plot = 1
        fig, axes = plt.subplots(n_subplots,n_subplots, sharex=True, sharey=True, figsize=(15, 15)) # rows, columns 4, 3
        # colors = ["r", "b", "k", "y", "g", "m", "pink"]
        colors = cm.rainbow(np.linspace(0, 1, len(y)%n_subplots+1))
        fig.tight_layout() 
        fig.supxlabel("X coord (m)")
        fig.supylabel("Z coord (m)")
        fig.suptitle("Trajectory of each sequence\n centered at the beginning (t0)\n showing prediction and gt")
        out_centered = skeleton.transform_to_metric_space(out)
        y_centered = skeleton.transform_to_metric_space(y)
        x_centered = skeleton.transform_to_metric_space(x)
        xz_plane_out = np.stack([out_centered[...,0, 0], out_centered[...,0, 2]], axis=-1)
        xz_plane_y = np.stack([y_centered[...,0, 0], y_centered[...,0, 2]], axis=-1)
        xz_plane_x = np.stack([x_centered[...,0, 0], x_centered[...,0, 2]], axis=-1)
        i = 0
        track = 0
        for s, seq in enumerate(xz_plane_out):
            if track == n_tracks_per_plot and i!=n_subplots**2-1:
                track=0
                i += 1
            axes[i//n_subplots][i%n_subplots].plot(xz_plane_x[s][...,0], xz_plane_x[s][...,1], c='k')
            axes[i//n_subplots][i%n_subplots].plot(xz_plane_y[s][...,0], xz_plane_y[s][...,1], c='r')
            axes[i//n_subplots][i%n_subplots].plot(seq[...,0], seq[...,1], c=colors[track])
            track += 1
        plt.tight_layout() 
        # dir_path = os.path.join(output_log_path, f"images_{kwargs['split']}", )
        # os.makedirs(dir_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'Trajectory_out_sequences_centered_tmiddle.png')) 
               
    def drop_results(state, mode):
        with open(os.path.join(output_log_path, f"eval_{mode}.yaml"), mode="w", encoding="utf-8") as file:
            metrics = {k:v.tolist() if torch.is_tensor(v) else v for k,v in state.metrics.items()}
            yaml.dump(metrics, file)
        
        out = []
        gt = []
        inp = []
        for b, (model_out, y, x) in enumerate(state.output):
            out.append(model_out)
            gt.append(y)
            inp.append(x)      
        save_dir = os.path.join(output_log_path, f"images_{kwargs['split']}", )
        os.makedirs(save_dir, exist_ok=True)
        print_trajectories(torch.cat(out, dim=0).cpu(), torch.cat(gt, dim=0).cpu(), torch.cat(inp, dim=0).cpu(), save_dir)
       
        
        batch_idx = 0
        model_out, y, x = state.output[batch_idx]
        idx = [10, 20]
        # pick random or specifi samples
        print("Saving visual samples")
        for i in tqdm([16]): #range(len(model_out))
            dataset = dataset_train if mode=="train" else dataset_eval
            dir_path = os.path.join(save_dir, 
                    f"track_{h36m_dataset_eval['seq_names'][i].replace('/', '_')}_idx_{i}")
            # imgs_path = dataset._data["img_paths_in+out"][dataset._data["correspondence"][batch_idx*batch_size_eval+i]]
            os.makedirs(dir_path, exist_ok=True)
            save_visuals(model_out, y, x, idx=i, save_path=dir_path, mode=mode, imgs_path=None)
            print_trajectories(model_out[i].unsqueeze(0).cpu(), y[i].unsqueeze(0).cpu(), 
                           x[i].unsqueeze(0).cpu(), dir_path)
    
    state = evaluator.run(data_loader_eval)
    state.metrics["VIM avg"] = state.metrics["VIM"].mean()
    state.metrics["VIM table"] = [round((state.metrics["VIM"][i]).item(), 1) for i in [1, 24, 49, 74, -1]]
    state.metrics["MPJPE table"] = [round((state.metrics["MPJPE"][i]).item(), 1) for i in [1, 24, 49, 74, -1]]
    state.metrics["MPJPE avg"] = round(state.metrics["MPJPE"].mean().item(), 1)
    state.metrics["MPJPE POSE table"] = [round((state.metrics["MPJPE_POSE"][i]).item(), 1) for i in [1, 24, 49, 74, -1]]
    state.metrics["MPJPE POSE avg"] = round(state.metrics["MPJPE_POSE"].mean().item(), 1)
    state.metrics["MPJPE HIP table"] = [round((state.metrics["MPJPE_HIP"][i]).item(), 1) for i in [1, 24, 49, 74, -1]]
    state.metrics["MPJPE HIP avg"] = round(state.metrics["MPJPE_HIP"].mean().item(), 1)
    
    state.metrics["MPJPE avg@75"] = round(state.metrics["MPJPE"][:75].mean().item(), 1)
    state.metrics["MPJPE POSE avg@75"] = round(state.metrics["MPJPE_POSE"][:75].mean().item(), 1)
    state.metrics["MPJPE HIP avg@75"] = round(state.metrics["MPJPE_HIP"][:75].mean().item(), 1)
    

    state.metrics["LIMB dist GT per joint"] = state.metrics["LIMB dist GT"].mean(0)
    state.metrics["LIMB dist GT per joint table"] = [round((state.metrics["LIMB dist GT per joint"][i]).item(), 1) for i in range(len(state.metrics["LIMB dist GT per joint"]))]
    state.metrics["LIMB dist GT per joint table sorted"] = sorted({limb: dist for limb, dist in zip(skeleton.limbseq_names, state.metrics["LIMB dist GT per joint table"])}.items(), key=lambda x:x[1])
    state.metrics["LIMB dist GT per joint table std"] = torch.std(state.metrics["LIMB dist GT per joint"])
    state.metrics["LIMB dist GT per time"] = state.metrics["LIMB dist GT"].mean(-1)
    state.metrics["LIMB dist GT per time table"] = [round((state.metrics["LIMB dist GT per time"][i]).item(), 1) for i in [1, 24, 49, 74, -1]]
    state.metrics["LIMB dist GT avg"] = round(state.metrics["LIMB dist GT"].mean().item(), 1)

    print(f"Result on eval {kwargs['split']}: ", state.metrics)
    drop_results(state, mode=kwargs['split'])
    
    

    
import argparse

parser = argparse.ArgumentParser(description='SomoF Download')

parser.add_argument('--load_path',
                    type=str,
                    help='The path for loading',
                    default='./output/somof/exp_name/checkpoints/36000.pth.tar')


parser.add_argument('--device',
                    type=str,
                    help='Training Device.',
                    default='cuda')

parser.add_argument('--split',
                    type=str,
                    help='Split. val or test',
                    default='val')

args = parser.parse_args()

if __name__ == '__main__':
    output_path = os.path.dirname(os.path.dirname(args.load_path))  #'./output/hip_extra_node/weight_-6_epoch16/checkpoints/36000.pth.tar'
    with open(os.path.join(output_path, "config.yaml"), 'r') as stream:
        params = yaml.safe_load(stream)
        
    params['device'] = args.device
    params['load_path'] = args.load_path
    params['split'] = args.split

    load_epoch = int(params['load_path'].split("/")[-1].replace('.pth.tar', ''))
    params['output_log_path'] = os.path.join(output_path, f"eval_{load_epoch}")
    os.makedirs(params['output_log_path'], exist_ok=True)
    eval(**params)