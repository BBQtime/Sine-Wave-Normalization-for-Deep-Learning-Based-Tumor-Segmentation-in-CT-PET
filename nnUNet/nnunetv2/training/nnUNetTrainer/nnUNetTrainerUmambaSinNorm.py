
from typing import Tuple, Union, List

import numpy as np
import torch

from torch import autocast, nn

import pydoc

from time import time
from os.path import join

#from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.resnet_sin_normal import ResidualEncoderUMamba

class nnUNetTrainerUmambaSinNorm(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Gradient accumulation setup
        # Gradient accumulation setup
        self.accumulation_steps = 16  # Number of batches to accumulate gradients over
        self.current_step = 0  #init current step for each epoch
        self.num_epochs = 1000
        self.new_num_epochs = 1200
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        
    def train_step(self, batch: dict) -> dict:
        # Initialize grad_norm to None
        grad_norm = None
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # No need to zero grad every step, it's handled after accumulation
        # self.optimizer.zero_grad(set_to_none=True)

        # Autocast only for CUDA
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            # Scale loss, backward and unscale
            self.grad_scaler.scale(loss).backward()
            if (self.current_step + 1) % self.accumulation_steps == 0:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)  
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # Reset gradients after the accumulation step
        else:
            loss.backward()
            if (self.current_step + 1) % self.accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)   
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # Reset gradients after the accumulation step

        # Log large gradients if present
        if grad_norm is not None and grad_norm > 100:
            self.print_to_log_file(f"->>> extreme Large gradient norm: {grad_norm}")

        # Increment current step after processing a batch
        self.current_step += 1

        return {'loss': loss.detach().cpu().numpy()}

    def on_epoch_end(self):
        self.current_step = 0
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        
    def run_training(self):
        self.on_train_start()
        end_epoch = self.num_epochs
        if self.new_num_epochs != self.num_epochs:
            end_epoch = self.new_num_epochs

        for epoch in range(self.current_epoch, end_epoch):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
        
def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                        allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
    #chose your network here: ResidualEncoderUNet, ResidualUNet, ResidualEncoderUMamba
    nw_class = ResidualEncoderUMamba

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision
    print(architecture_kwargs)
    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

