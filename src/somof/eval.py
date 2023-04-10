import os
import yaml
from typing import Sequence
import numpy as np

import torch
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, EpochOutputStore
from ignite.engine import Events, Engine, DeterministicEngine
from ignite.metrics import Loss

import sys
sys.path.append(os.path.abspath('./'))

from skeleton import SoMoFSkeleton
from dataset.somof_dataset import SoMoFDataset
from dataset.somof_torch_dataset import SoMoFTorchDataset
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
          seed: int = 60000,
          detect_anomaly: bool = False,
          **kwargs) -> None:
    """
    Evaluate
    """
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)
    
    # Load skeleton configuration
    skeleton = SoMoFSkeleton(num_joints=kwargs["num_joints"], if_consider_hip=kwargs["if_consider_hip"], pose_box_size=kwargs["pose_box_size"])

    # Create model
    model = Motion(num_nodes=skeleton.num_nodes,
                   T=skeleton.nodes_type_id, **kwargs).to(device)

    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Load the SoMoF Dataset from disk
    prediction_horizon_train = kwargs['prediction_horizon_train']
    prediction_horizon_eval = kwargs['prediction_horizon_eval']

    # Create TorchDatasets for training
    # dataset_train = SoMoFTorchDataset(SoMoFDataset(split="train", **kwargs), **kwargs)
    # data_loader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size_eval,
    #                                num_workers=0)    
    
    # Create TorchDatasets for validation
    dataset_eval = SoMoFTorchDataset(SoMoFDataset(split="valid", **kwargs), **kwargs)
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
    
    def rescale_output_no_hip(output):
        output = [skeleton.rescale_to_hip_box(out[...,len(skeleton.node_hip):,:]) for out in output[:2]]
        return output[0], output[1]
    
    # Define ignite metrics
    mpjpe = MeanPerJointPositionError(output_transform=rescale_output)
    mpjpe_pose = MeanPerJointPositionError(output_transform=rescale_output_no_hip)
    mpjpe_avg = MeanPerJointPositionError(output_transform=rescale_output, keep_time_dim=False)
    mpjpe_avg_pose = MeanPerJointPositionError(output_transform=rescale_output_no_hip, keep_time_dim=False)

    fde = FinalDisplacementError(output_transform=rescale_output)
    fde_pose = FinalDisplacementError(output_transform=rescale_output_no_hip)
    vim = VIM(output_transform=rescale_output_no_hip, dataset_name="3dpw")


    loss_metric_pose = Loss(model.loss_pose, output_transform=lambda x: (x[0], x[1]))
    loss_metric_hip = Loss(model.loss_hip, output_transform=lambda x: (x[0], x[1]))


    evaluator = DeterministicEngine(validation_step_eval)
    evaluator.add_event_handler(Events.ITERATION_STARTED, preprocess)
    mpjpe.attach(evaluator, 'MPJPE')
    mpjpe_pose.attach(evaluator, 'MPJPE_POSE')
    mpjpe_avg.attach(evaluator, 'MPJPE_AVG')
    mpjpe_avg_pose.attach(evaluator, 'MPJPE_AVG_POSE')
    loss_metric_pose.attach(evaluator, 'Loss Pose')
    loss_metric_hip.attach(evaluator, 'Loss Hip')
    fde.attach(evaluator, 'FDE')
    fde_pose.attach(evaluator, 'FDE_POSE')
    vim.attach(evaluator, 'VIM')

    
    eos = EpochOutputStore()
    eos.attach(evaluator, 'output')

    pbar = ProgressBar()
    pbar.attach(evaluator)
    
    def save_visuals(model_out, y, x, idx=-1, mode="train", imgs_path=None):
        image_dict = visuals_step_3D_hc(skeleton, x[idx].cpu(), model_out[idx].cpu(), y[idx].cpu(), imgs_path=imgs_path)
        for name, image in image_dict.items():
            dir_path = os.path.join(output_log_path, f"images")
            os.makedirs(dir_path, exist_ok=True)
            fpath = os.path.join(dir_path, f"{mode}_img_{idx}_{name}")
            save_img(image, fpath) if "gif" not in name else save_gif(image, name=fpath, fps=5)
            
    def drop_results(state, mode):
        with open(os.path.join(output_log_path, f"eval_{mode}.yaml"), mode="w", encoding="utf-8") as file:
            metrics = {k:v.tolist() if torch.is_tensor(v) else v for k,v in state.metrics.items()}
            yaml.dump(metrics, file)
        idx = [10, 20]
        batch_idx = 0
        model_out, y, x = state.output[batch_idx]
        # pick random samples
        for i in idx:
            dataset = dataset_train if mode=="train" else dataset_eval
            imgs_path = dataset._data["img_paths_in+out"][dataset._data["correspondence"][batch_idx*batch_size_eval+i]]
            save_visuals(model_out, y, x, idx=i, mode=mode, imgs_path=imgs_path)
    
    state = evaluator.run(data_loader_eval)
    state.metrics["VIM avg"] = state.metrics["VIM"].mean()
    state.metrics["VIM table"] = [round((state.metrics["VIM"][i]/10).item(), 1) for i in [1,3,7,9,13]] # 100ms, 240ms, 500ms, 640ms, 900ms
    state.metrics["VIM table Overall"] = round(np.array(state.metrics["VIM table"]).mean().item(), 1)
    state.metrics["MPJPE table"] = [round((state.metrics["MPJPE"][i]/10).item(), 1) for i in [1,3,7,9,13]]
    state.metrics["MPJPE table Overall"] = round(np.array(state.metrics["MPJPE table"]).mean().item(), 1)

    print(f"Result on eval val: ", state.metrics)
    drop_results(state, mode="valid")

    

    
    
    
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

args = parser.parse_args()

if __name__ == '__main__':
    output_path = os.path.dirname(os.path.dirname(args.load_path))
    with open(os.path.join(output_path, "config.yaml"), 'r') as stream:
        params = yaml.safe_load(stream)
        
    params['device'] = args.device
    params['load_path'] = args.load_path

    load_epoch = int(params['load_path'].split("/")[-1].replace('.pth.tar', ''))
    params['output_log_path'] = os.path.join(output_path, f"eval_{load_epoch}")
    os.makedirs(params['output_log_path'], exist_ok=True)
    eval(**params)