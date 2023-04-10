import os
import yaml
from typing import Sequence
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, EpochOutputStore
from ignite.engine import Events, Engine, DeterministicEngine

import sys
sys.path.append(os.path.abspath('./'))

from skeleton import SoMoFSkeleton
from dataset.somof_test_dataset import SoMoFTestDataset
from dataset.somof_torch_test_dataset import SoMoFTorchTestDataset
from model.utils.torch import set_seed
from model.network import Motion
from model.utils.visuals import get_color_frames_in_out
from model.utils.image import  save_gif
from model.utils.image import load_image
                
                
def test(output_log_path: str,
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

    checkpoint = torch.load(kwargs['load_path'])
    model.load_state_dict(checkpoint, strict=True)
    
    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    
    prediction_horizon_eval = kwargs['prediction_horizon_eval']

    # Create TorchDatasets for testing
    dataset_test = SoMoFTorchTestDataset(SoMoFTestDataset(**kwargs), **kwargs)
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size_eval,
                                   num_workers=0)    


    # Define pre-process function to transform input to correct data type
    def preprocess(engine: Engine):
        engine.state.batch =  [t.to(device) for t in engine.state.batch[:]]
        
    def step_eval(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            x, seq_start = batch
            model_out, _, _ = model(x, prediction_horizon_eval, None)
            return model_out, seq_start
        
    def save_visual(data_track, name="frames"):
        track= data_track.clone()
        track[..., 1] *= -1
        frames = torch.stack(get_color_frames_in_out(track.cpu().numpy(), n_in_frames=dataset_test.history_length, 
                        limbseq=skeleton.limbseq, left_right_limb=skeleton.left_right_limb, 
                        xyz_range=None, center_pose=False, units="mm", 
                        as_tensor=True, orientation_like="h36m", title="Output Kpts"), dim=0)
        save_gif(frames, name=os.path.join(output_log_path, name), fps=5)
        
    evaluator = DeterministicEngine(step_eval)
    evaluator.add_event_handler(Events.ITERATION_STARTED, preprocess)
    
    eos = EpochOutputStore()
    eos.attach(evaluator, 'output')

    pbar = ProgressBar()
    pbar.attach(evaluator)
    
    
    state = evaluator.run(data_loader_test)
    output_kpts = torch.cat([kpts for kpts, start in state.output], dim=0)
    seq_start = torch.cat([start for kpts, start in state.output], dim=0)
    output = skeleton.transform_to_test_space(output_kpts, seq_start)
    output = output.view((-1, 2, 14, 13*3))
    output.shape
    with open(os.path.join(output_log_path, '3dpw_predictions.json'), 'w') as f:
        f.write(json.dumps(output.tolist()))
    
    if  kwargs["save_visual_sample"]: 
        print("Predictions dumped to json. Saving visual sample...")
        sample_idx = 0
        seqs = torch.cat([dataset_test._data["input"].to(output.device), output], dim=-2).view(-1, 30, 13, 3)
        seq = seqs[sample_idx]*1000
        save_visual(seq, name="frames")
        pose = seq - seq[..., 0,:].unsqueeze(-2)
        save_visual(pose, name="frames_pose")
        imgs_path = dataset_test._data["img_paths"][sample_idx]
        rgb_seq = np.array([load_image(img_path) for img_path in imgs_path])
        save_gif(rgb_seq, name=os.path.join(output_log_path, "rgb"), fps=5)

    
    print("Done!")
    
    
import argparse

parser = argparse.ArgumentParser(description='SomoF Test')

parser.add_argument('--load_path',
                    type=str,
                    help='The path for loading',
                    default='./output/somof/exp_name/checkpoints/36000.pth.tar')

parser.add_argument('--device',
                    type=str,
                    help='Training Device.',
                    default='cuda')

parser.add_argument('--save_visual_sample',
                    default=False, 
                    action='store_true',
                    help='if saving a visual sample'
                    )

args = parser.parse_args()

if __name__ == '__main__':
    output_path = os.path.dirname(os.path.dirname(args.load_path))
    with open(os.path.join(output_path, "config.yaml"), 'r') as stream:
        params = yaml.safe_load(stream)
        
    params['device'] = args.device
    params['load_path'] = args.load_path
    params['save_visual_sample'] = args.save_visual_sample

    load_epoch = int(params['load_path'].split("/")[-1].replace('.pth.tar', ''))
    params['output_log_path'] = os.path.join(output_path, f"test_{load_epoch}")
    os.makedirs(params['output_log_path'], exist_ok=True)
    test(**params)