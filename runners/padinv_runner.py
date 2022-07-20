# python3.7
"""Contains the runner for GAN Inversion with PadInv Encoder."""

import os
from copy import deepcopy
import torch

from .base_runner import BaseRunner

__all__ = ['PadInvRunner']


class PadInvRunner(BaseRunner):
    """Defines the runner for encoder training."""

    def __init__(self, config):
        super().__init__(config)
        # Load pretrained generator.
        self.load(filepath=self.config.data.pretrained_gan_path,
                  running_metadata=False,
                  learning_rate=False,
                  optimizer=False,
                  running_stats=False,
                  avg_latent=True)

    def load(self,
             filepath,
             running_metadata=True,
             optimizer=True,
             learning_rate=True,
             loss=True,
             augment=True,
             running_stats=False,
             avg_latent=True,
             map_location='cpu'):
        """Loads previous running status.

        This implementation overrides original `load()` in `BaseRunner`, with
        additional support on loading average W latent for encoder training.

        Args:
            filepath: File path to load the checkpoint.
            running_metadata: Whether to load the running metadata, such as
                batch size, current iteration, etc. (default: True)
            optimizer: Whether to load the optimizer. (default: True)
            learning_rate: Whether to load the learning rate. (default: True)
            loss: Whether to load the loss. (default: True)
            augment: Whether to load the augmentation, especially the adaptive
                augmentation probability. (default: True)
            running_stats: Whether to load the running stats. (default: False)
            avg_latent: Whether to load average W latent for encoder training.
                        (default: True)
            map_location: Map location used for model loading. (default: `cpu`)
        """
        self.logger.info(f'Resuming from checkpoint `{filepath}` ...')
        if not os.path.isfile(filepath):
            raise IOError(f'Checkpoint `{filepath}` does not exist!')
        map_location = map_location.lower()
        assert map_location in ['cpu', 'gpu']
        if map_location == 'gpu':
            map_location = lambda storage, location: storage.cuda(self.device)
        checkpoint = torch.load(filepath, map_location=map_location)
        # Load models.
        if 'models' not in checkpoint:
            checkpoint = {'models': checkpoint}
        for model_name, model in self.models.items():
            # Load the pre-trained generator (smoothed version first).
            if model_name == 'generator':
                if checkpoint['models']['generator_smooth']:
                    state_dict = checkpoint['models']['generator_smooth']
                elif checkpoint['models']['generator']:
                    state_dict = checkpoint['models']['generator']
                else:
                    self.logger.warning('Model `generator` is not included in '
                                        'the checkpoint, and hence will NOT be '
                                        'loaded!', indent_level=1)
                    continue
            # Load pre-trained models except the generator.
            else:
                if model_name not in checkpoint['models']:
                    self.logger.warning(f'Model `{model_name}` is not included '
                                        'in the checkpoint, and hence will NOT '
                                        'be loaded!', indent_level=1)
                    continue
                state_dict = checkpoint['models'][model_name]
            # Load state dict.
            model.load_state_dict(state_dict)
            self.logger.info(f'Successfully loaded model `{model_name}`.',
                             indent_level=1)
        # Load running metadata.
        if running_metadata:
            if 'running_metadata' not in checkpoint:
                self.logger.warning('Running metadata is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self._iter = checkpoint['running_metadata']['iter']
                self._start_iter = self._iter
                self.seen_img = checkpoint['running_metadata']['seen_img']
        # Load optimizers.
        if optimizer:
            if 'optimizers' not in checkpoint:
                self.logger.warning('Optimizers are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for opt_name, opt in self.optimizers.items():
                    if opt_name not in checkpoint['optimizers']:
                        self.logger.warning(f'Optimizer `{opt_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    opt.load_state_dict(checkpoint['optimizers'][opt_name])
                    self.logger.info(f'Successfully loaded optimizer '
                                     f'`{opt_name}`.', indent_level=1)
        # Load learning rates.
        if learning_rate:
            if 'learning_rates' not in checkpoint:
                self.logger.warning('Learning rates are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for lr_name, lr in self.lr_schedulers.items():
                    if lr_name not in checkpoint['learning_rates']:
                        self.logger.warning(f'Learning rate `{lr_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    lr.load_state_dict(checkpoint['learning_rates'][lr_name])
                    self.logger.info(f'Successfully loaded learning rate '
                                     f'`{lr_name}`.', indent_level=1)
        # Load loss.
        if loss:
            if 'loss' not in checkpoint:
                self.logger.warning('Loss is not included in the checkpoint, '
                                    'and hence will NOT be loaded!',
                                    indent_level=1)
            else:
                self.loss.load_state_dict(checkpoint['loss'])
                self.logger.info('Successfully loaded loss.', indent_level=1)
        # Load augmentation.
        if augment:
            if 'augment' not in checkpoint:
                self.logger.warning('Augmentation is not included in '
                                    'the checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.augment.load_state_dict(checkpoint['augment'])
                self.logger.info('Successfully loaded augmentation.',
                                 indent_level=1)
        # Load running stats.
        #  Only resume `stats_pool` from checkpoint.
        if running_stats:
            if 'running_stats' not in checkpoint:
                self.logger.warning('Running stats is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.running_stats.stats_pool = deepcopy(
                    checkpoint['running_stats'])
                self.running_stats.is_resumed = True  # avoid conflicts when add
                self.logger.info('Successfully loaded running stats.',
                                 indent_level=1)

        # Load the average latent code (mainly for encoder training).
        if avg_latent:
            if 'w_avg' not in checkpoint['models']['generator']:
                # TODO: We can sample enormous w latents to estimate w_avg even
                #  thought it is not provided.
                self.logger.warning('W average latent is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!')
            else:
                w_avg_latent = checkpoint['models']['generator']['w_avg']
                w_avg_latent = w_avg_latent.to(device=self.device)
                self.models['encoder'].avg_latent = w_avg_latent
                self.logger.info('Successfully loaded W average latent.',
                                 indent_level=1)

        # Log message.
        tailing_message = ''
        if running_metadata and 'running_metadata' in checkpoint:
            tailing_message = f' (iteration {self.iter})'
        self.logger.info(f'Successfully loaded from checkpoint `{filepath}`.'
                         f'{tailing_message}')

    def build_models(self):
        super().build_models()
        self.models['generator'].requires_grad_(False)

    def train_step(self, data):
        self.models['encoder'].requires_grad_(True)
        if self.config.use_discriminator:
            self.models['discriminator'].requires_grad_(False)

        E_loss = self.loss.E_loss(self, data, sync=True)
        self.zero_grad_optimizer('encoder')
        E_loss.backward()
        self.step_optimizer('encoder')

        if self.config.use_discriminator:
            self.models['encoder'].requires_grad_(False)
            self.models['discriminator'].requires_grad_(True)
            D_loss = self.loss.D_loss(self, data, sync=True)
            self.zero_grad_optimizer('discriminator')
            D_loss.backward()
            self.step_optimizer('discriminator')
