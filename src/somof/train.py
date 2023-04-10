import os
import yaml
from typing import Sequence
from functools import partial
import numpy as np

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler, EpochOutputStore
from ignite.engine import Events, Engine, DeterministicEngine
from ignite.metrics import RunningAverage, Loss
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.handlers import Checkpoint, DiskSaver

import sys
sys.path.append(os.path.abspath('./'))

from skeleton import SoMoFSkeletonUnitNorm, SoMoFSkeletonVelocity
from dataset.somof_dataset import SoMoFDataset
from dataset.somof_torch_dataset import SoMoFTorchDataset
from dataset.d3pw_torch_dataset import D3PW_SoMoFTorchDataset
from dataset.d3pw_dataset import D3PW_SoMoFDataset
from model.utils.torch import set_seed
from model.utils.tensorboard import set_default_tb_train_logging
from model.network import Motion
from model.core.metrics.mpjpe import MeanPerJointPositionError
from model.core.metrics.fde import FinalDisplacementError
from model.core.metrics.vim import VIM
from model.utils.visuals import visuals_step_3D_hc
from model.utils.image import save_img, save_gif




def train(output_log_path: str,
          lr: float,
          batch_size: int,
          num_epochs: int,
          eval_frequency: int = None,
          random_prediction_horizon: bool = False,
          curriculum_it: int = 0,
          batch_size_eval: int = 0,
          num_iteration_eval: int = 0,
          clip_grad_norm: float = None,
          device: str = 'cpu',
          seed: int = 52345,
          num_workers: int = 0,
          detect_anomaly: bool = False,
          **kwargs) -> None:
    """
    Trains Motion on H3.6M Dataset
    """
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)

    # Load skeleton configuration
    if kwargs["hip_mode"] == "norm":
        skeleton = SoMoFSkeletonUnitNorm(**kwargs)
    elif kwargs["hip_mode"] == "velocity": 
        skeleton = SoMoFSkeletonVelocity(**kwargs)
    else: 
        assert 0, "Mode non iplemented"    

    # # Create model
    model = Motion(num_nodes=skeleton.num_nodes,
                   T=skeleton.nodes_type_id, **kwargs).to(device)

    print(f"Created Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Load the SoMoF Dataset from disk
    
    prediction_horizon_train = kwargs['prediction_horizon_train']
    prediction_horizon_eval = kwargs['prediction_horizon_eval']

    # Create TorchDatasets for training
    if kwargs["pretrain_somof"]:
        Dataset = D3PW_SoMoFDataset
        TorchDataset = D3PW_SoMoFTorchDataset
    else: 
        Dataset = SoMoFDataset
        TorchDataset = SoMoFTorchDataset
    
    tranforms = []
    if kwargs["if_random_rotation"]:
        tranforms.append(skeleton.rotate_around_y_axis)
    if kwargs["if_rescale"]:
        tranforms.append(skeleton.rescale_body)
        
    def transform(transform_list, kpts):
        for t in transform_list:
            kpts = t(kpts)
        return kpts
        
    dataset_train = TorchDataset(Dataset(split="train", skeleton=skeleton, **kwargs), 
                                 transform=partial(transform, tranforms),**kwargs)
    dataset_train.mirror()  # Dataset Augmentation mirroring left and right
    if kwargs["if_reverse"]:
        dataset_train.reverse() 
    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True)    
    
    # Create TorchDatasets for and validation
    dataset_eval = SoMoFTorchDataset(SoMoFDataset(split="valid", skeleton=skeleton, **kwargs), **kwargs)
    data_loader_eval = DataLoader(dataset_eval, shuffle=False, batch_size=batch_size_eval, num_workers=0)
    
    dataset_eval_train = SoMoFTorchDataset(SoMoFDataset(split="train", skeleton=skeleton, **kwargs), **kwargs)
    data_loader_train_eval = DataLoader(dataset_eval_train, shuffle=True, batch_size=batch_size_eval, num_workers=0)
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    class CurriculumLearning:
        def __init__(self):
            self.param_groups = [{'curriculum_factor': 0.}]

    curriculum = CurriculumLearning()

    if curriculum_it is not None and curriculum_it > 0.0:
        curriculum_scheduler = CosineAnnealingScheduler(curriculum,
                                                        'curriculum_factor',
                                                        start_value=1.0,
                                                        end_value=0.,
                                                        cycle_size=curriculum_it,
                                                        start_value_mult=0.0,
                                                        save_history=True,
                                                        )

    # Define pre-process function to transform input to correct data type
    def preprocess(engine: Engine):
        engine.state.batch =  [t.to(device) for t in engine.state.batch[:]]

    # Define process function which is called during every training step
    def train_step(engine: Engine, batch: Sequence[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        x, y = batch

        ph = max(int(np.rint((1. - curriculum.param_groups[0]['curriculum_factor']) * prediction_horizon_train)), 1)
        if ph > 1 and random_prediction_horizon:
            ph = np.random.randint(1, ph)
        y = y[:, :ph]
        p_pred, _, kwargs = model(x, ph, None)

        loss_pose_unscaled = model.loss_pose(p_pred, y)
        loss_hip_unscaled = model.loss_hip(p_pred, y)

        loss = loss_pose_unscaled*1e-1 +  loss_hip_unscaled

        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        return loss, loss_pose_unscaled, loss_hip_unscaled, p_pred, y, ph

    def validation_step(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            x, y = batch
            model_out, _, _ = model(x, prediction_horizon_eval, None)
            return model_out, y, x
        
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
    vim = VIM(output_transform=rescale_output_no_hip, dataset_name="3dpw")

    fde = FinalDisplacementError(output_transform=rescale_output)
    fde_pose = FinalDisplacementError(output_transform=rescale_output_no_hip)

    loss_metric_pose = Loss(model.loss_pose, output_transform=lambda x: (x[0], x[1]))
    loss_metric_hip = Loss(model.loss_hip, output_transform=lambda x: (x[0], x[1]))


    # Define Training, Evaluation and Test Engines and attach metrics
    trainer = DeterministicEngine(train_step)

    objects_to_checkpoint = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'curriculum_scheduler': curriculum_scheduler} #'lr_scheduler': lr_scheduler, 
    checkpoint_handler = Checkpoint(objects_to_checkpoint, DiskSaver(os.path.join(output_log_path, "checkpoints"), create_dir=True, require_empty=False))
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=eval_frequency), checkpoint_handler) 
    if kwargs['load']:
        # checkpoint = torch.load(kwargs['load_path'])
        checkpoint = torch.load(kwargs['load_path'])
        # model.load_state_dict(checkpoint)#, strict=False)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint) 


    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'Loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'Loss Pose')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'Loss Hip')

    trainer.add_event_handler(Events.ITERATION_STARTED, preprocess)
    if curriculum_it is not None and curriculum_it > 0.0:
        trainer.add_event_handler(Events.ITERATION_STARTED, curriculum_scheduler)

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
    mpjpe_avg.attach(evaluator, 'MPJPE_AVG')

    # Evaluate on Train dataset
    evaluator_train = DeterministicEngine(validation_step)
    evaluator_train.add_event_handler(Events.ITERATION_STARTED, preprocess)
    mpjpe.attach(evaluator_train, 'MPJPE')
    mpjpe_pose.attach(evaluator_train, 'MPJPE_POSE')
    mpjpe_avg.attach(evaluator_train, 'MPJPE_AVG')
    mpjpe_avg_pose.attach(evaluator_train, 'MPJPE_AVG_POSE')

    fde.attach(evaluator_train, 'FDE')
    fde_pose.attach(evaluator_train, 'FDE_POSE')
    vim.attach(evaluator_train, 'VIM')
    loss_metric_pose.attach(evaluator_train, 'Loss Pose')
    loss_metric_hip.attach(evaluator_train, 'Loss Hip')


    testers = dict()
    # if kwargs['dataset_downsample_factor'] > 1:
    #     for action in kwargs['test_actions'] + ['average']:
    #         testers[action] = Engine(validation_step)
    #         mpjpe.attach(testers[action], 'MPJPE')
    #         me.attach(testers[action], 'EUCLIDEAN')
    #         testers[action].add_event_handler(Events.ITERATION_STARTED, preprocess)

    # Setup tensorboard logging and progressbar for training
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, 'tb'))
    setup_logging(tb_logger, trainer, evaluator, evaluator_train, optimizer, model, testers, eval_frequency)

    pbar = ProgressBar()
    pbar.attach(trainer, ['Loss']),
    pbar.attach(evaluator),
    pbar.attach(evaluator_train)
    
    # @trainer.on(Events.ITERATION_COMPLETED(every=10))#eval_frequency))
    # def train_iter_completed(engine: Engine):

    #     # Run evaluation on random chunk of dataset
    #     evaluator.run(data_loader_eval)
    #     model_out, y, x = evaluator.state.output[0]
    #     image_dict = visuals_step_3D_hc(skeleton, x[0].cpu(), model_out[0].cpu(), y[0].cpu())
    #     for name, image in image_dict.items():
    #         dir_path = os.path.join(output_log_path, "images", "Iter_{:09d}".format(engine.state.iteration))
    #         os.makedirs(dir_path, exist_ok=True)
    #         fpath = os.path.join(dir_path, name)
    #         save_img(image, fpath) if "gif" not in name else save_gif(image, name=fpath, fps=5)
                
    #Setup evaluation process between epochs
    if eval_frequency is not None:
        @trainer.on(Events.ITERATION_COMPLETED(every=eval_frequency))#eval_frequency))
        def train_epoch_completed(engine: Engine):
            # for action, tester in testers.items():
                # dataset_test_action = H36MTorchDataset(h36m_dataset,
                #                                       action=action,
                #                                       subjects=kwargs['subjects_test'],
                #                                       num_samples=256,
                #                                       no_overlapping_samples=True, 
                #                                       sampling_dist=kwargs['dataset_downsample_factor'],
                #                                       **kwargs)
                # tester.run(DataLoader(dataset_test_action, batch_size=256))

            # Run evaluation on random chunk of dataset
            evaluator.run(data_loader_eval)
            model_out, y, x = evaluator.state.output[0]
            image_dict = visuals_step_3D_hc(skeleton, x[-1].cpu(), model_out[-1].cpu(), y[-1].cpu())
            for name, image in image_dict.items():
                dir_path = os.path.join(output_log_path, "images", "Epoch_{:09d}".format(engine.state.epoch))
                os.makedirs(dir_path, exist_ok=True)
                fpath = os.path.join(dir_path, name)
                save_img(image, fpath) if "gif" not in name else save_gif(image, name=fpath.replace("gif", ""), fps=5)
            
            evaluator_train.run(data_loader_train_eval, epoch_length=num_iteration_eval)
            torch.save(model.state_dict(), os.path.join(output_log_path, f"checkpoints/{engine.state.epoch}.pth.tar"))

    trainer.run(data_loader_train, max_epochs=num_epochs)

    tb_logger.close()


def setup_logging(tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, evaluator_train, optimizer: Optimizer,
                  model: torch.nn.Module, testers: dict = None, eval_frequency=1):
    set_default_tb_train_logging(tb_logger, trainer, optimizer, model, if_consider_hip=True)

    tb_custom_scalar_layout = {}

    #　We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    
    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "losses/train",
           'metric_names': ["Loss Pose", "Loss Hip"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "losses/validation",
           'metric_names': ["Loss Pose", "Loss Hip"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )
        
        
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "validation",
           'metric_names': ["Loss Pose", "Loss Hip", "VIM", "FDE", "FDE_POSE", "MPJPE_AVG", "MPJPE_AVG_POSE", "MPJPE", "MPJPE_POSE"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MPJPE': ['Multiline', [rf"validation/MPJPE/.*"]],
                'MPJPE_POSE': ['Multiline', [rf"validation/MPJPE_POSE/.*"]]
            }
        }
    }

    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "validation_train",
           'metric_names': ["Loss Pose", "Loss Hip", "VIM", "FDE", "FDE_POSE", "MPJPE_AVG", "MPJPE_AVG_POSE", "MPJPE",  "MPJPE_POSE"],
           'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
           }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MPJPE_T': ['Multiline', [rf"validation_train/MPJPE_T/.*"]],
                'MPJPE_POSE_T': ['Multiline', [rf"validation/MPJPE_POSE_T/.*"]]
            }
        }
    }

    if testers is not None:
        tb_cs_layout_mpjpe_t = {}
        for action, tester in testers.items():
            tb_logger.attach_output_handler(
                tester,
                event_name=Events.COMPLETED,
                **{'tag': f"test/{action}",
                   'metric_names': ["Loss Pose", "Loss Hip", "VIM", "FDE", "FDE_POSE", "MPJPE_AVG", "MPJPE_AVG_POSE", "MPJPE", "MPJPE_POSE"],
                   'global_step_transform': lambda engine, event_name: trainer.state.iteration // eval_frequency
                   }
            )
            tb_cs_layout_mpjpe_t[f"{action}_MPJPE"] = ['Multiline', [rf"test/{action}/MPJPE/.*"]]
            tb_cs_layout_mpjpe_t[f"{action}_MPJPE_POSE"] = ['Multiline', [rf"test/{action}/MPJPE_POSE/.*"]]

        tb_custom_scalar_layout = {
            **tb_custom_scalar_layout,
            **{
               'Test Actions MPJPE': tb_cs_layout_mpjpe_t,
               'Test Actions MPJPE_POSE': tb_cs_layout_mpjpe_t
               }
        }
    tb_logger.writer.add_custom_scalars(tb_custom_scalar_layout)


import argparse
import shutil
import datetime
from model.utils.tensorboard import get_tensorb_style_filename

parser = argparse.ArgumentParser(description='SomoF Download')
parser.add_argument('--config',
                    type=str,
                    help='The config file for training',
                    default='./config/somof.yaml')

parser.add_argument('--load',
                    default=False, 
                    action='store_true',
                    help='if loading the model')

parser.add_argument('--load_path',
                    type=str,
                    help='The path for loading',
                    default='./output/h36m/exp_name/checkpoints/36000.pth.tar')

parser.add_argument('--info',
                    type=str,
                    help='Additional information which will be added to the output path',
                    default='')

parser.add_argument('--debug',
                    type=bool,
                    help='Debug Flag. No atrifacts will be saved to disk.',
                    default=False)

parser.add_argument('--device',
                    type=str,
                    help='Training Device.',
                    default='cuda')

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
    params['info'] = args.info
    params['debug'] = args.debug
    params['device'] = args.device
    params['load'] = args.load
    params['load_path'] = args.load_path
    
    if not params['debug']:
        if not params['load']:
            dt_str = get_tensorb_style_filename() #datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not params['info'] == '':
                info_str = f"_{params['info']}"
            else:
                info_str = ''
            params['output_log_path'] = os.path.join(params['output_path'], f"{dt_str}{info_str}")
            os.makedirs(os.path.join(params['output_log_path'], 'checkpoints'), exist_ok=True)
            os.makedirs(params['output_log_path'], exist_ok=True)
            with open(os.path.join(params['output_log_path'], 'config.yaml'), 'w') as config_file:
                yaml.dump(params, config_file)
            shutil.copytree('./model/', os.path.join(params['output_log_path'], 'code/model'))
            shutil.copytree('./somof/', os.path.join(params['output_log_path'], 'code/somof'))
        else: 
            output_path = os.path.dirname(os.path.dirname(params['load_path']))
            params['output_log_path'] = output_path
            assert os.path.dirname(args.config) ==  output_path
            params['num_epochs'] *= 2
    train(**params)

