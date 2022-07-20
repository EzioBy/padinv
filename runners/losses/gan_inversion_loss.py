# python3.7
"""Defines loss functions for GAN Inversion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model
from models.lpips.lpips import LPIPS
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['GANInversionLoss']


class GANInversionLoss(BaseLoss):
    """Contains the class to compute losses for GAN Inversion.

    Basically, this class contains the computation of mse, lpips,
    identity similarity and w-norm losses.
    """

    def __init__(self, runner, e_loss_kwargs=None, d_loss_kwargs=None):
        # Get kwargs.
        self.e_loss_kwargs = e_loss_kwargs or dict()
        self.mse_weight = self.e_loss_kwargs.get('mse_weight', 1.0)
        self.lpips_weight = self.e_loss_kwargs.get('lpips_weight', 0.8)
        self.idsim_weight = self.e_loss_kwargs.get('idsim_weight', 0.1)
        self.adv_weight = self.e_loss_kwargs.get('adv_weight', 0.1)
        self.wnorm_weight = self.e_loss_kwargs.get('wnorm_weight', 0)
        # lpips_net indicates the network architecture of LPIPS.
        # Note that we provide 2 versions of LPIPS implementation.
        self.lpips_net = self.e_loss_kwargs.get('lpips_net', 'alex')
        assert self.lpips_net in ['alex', 'vgg'], 'Currently we support alex and vgg as LPIPS backbones.'

        self.use_discriminator = False
        self.d_loss_kwargs = d_loss_kwargs or dict()
        if len(self.d_loss_kwargs) > 0:
            self.use_discriminator = True
            self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10)
            self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0)
            self.r1_interval = self.d_loss_kwargs.get('r1_interval', 16)
            self.r2_interval = self.d_loss_kwargs.get('r2_interval', 16)

        # Loss init including pretrained models in need.
        self.mse_loss = nn.MSELoss()
        if self.lpips_weight > 0:
            if self.lpips_net == 'alex':
                self.lpips = LPIPS(net_type='alex').cuda().eval()
            elif self.lpips_net == 'vgg':
                self.lpips = build_model('PerceptualModel')
        if self.idsim_weight > 0:
            self.arcface_model = build_model('ArcFaceModel')

        # Set runner logs.
        runner.running_stats.add('Loss/MSE',
                                 log_name='loss_mse',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/LPIPS',
                                 log_name='loss_lpips',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/IDLOSS',
                                 log_name='loss_id',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/ADVLOSS',
                                 log_name='loss_adv',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/WLOSS',
                                 log_name='loss_wnorm',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/E',
                                 log_name='loss_encoder',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')

        if self.use_discriminator:
            runner.running_stats.add('Loss/D_total',
                                     log_name='loss_discriminator',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')
            runner.running_stats.add('Loss/D',
                                     log_name='loss_real_fake_discriminate',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')
            runner.running_stats.add('Loss/GP',
                                     log_name='loss_gradient_penalty',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')

    @staticmethod
    def compute_idsim_loss(id_model, fake_imgs, real_imgs):
        """Computes identity similarity loss.

        This loss utilizes pretrained ArcFace model to extract identity-related
        features. And cosine similarity is used for computation.
        """
        real_feats = id_model(real_imgs).detach()
        fake_feats = id_model(fake_imgs)
        id_sim = torch.mean(torch.sum(real_feats.mul(fake_feats), dim=1))
        id_loss = 1 - id_sim

        return id_loss

    @staticmethod
    def compute_wnorm_loss(latent, latent_avg=None, start_from_latent_avg=True):
        """Computes wnorm loss.

        This loss forces the generating W+ latent code to be closer to the
        average, in order to generate more editable results.
        """
        if start_from_latent_avg:
            latent = latent - latent_avg
        return torch.mean(latent.norm(2, dim=(1, 2)))

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    @staticmethod
    def run_E(runner, input_images, sync=True):
        """Forward Encoder."""
        E = runner.ddp_models['encoder']
        with ddp_sync(E, sync=sync):
            return E(input_images)

    @staticmethod
    def run_G(runner, style_latents, padding_maps):
        """Forward the synthesis network of the pretrained generator."""
        G = runner.models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        images = G.synthesis(wp=style_latents,
                             padding_maps=padding_maps,
                             **G_kwargs)['image']
        # Resize images to 256 to calculate losses if needed.
        if G.resolution > 256:
            resize_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
            images = resize_pool(images)
        return images

    @staticmethod
    def run_D(runner, image, sync=True):
        """Forward Discriminator."""
        D = runner.ddp_models['discriminator']
        with ddp_sync(D, sync=sync):
            return D(image)['score']

    def E_loss(self, runner, data, sync=True):
        """Compute losses for the encoder.

        Now this loss is designed for the Inversion task, so the input
        real_images are also labels."""

        total_loss = 0.0
        real_images = data['image'].detach()
        latent_dict = self.run_E(runner, real_images, sync=sync)
        style_latents = latent_dict['style_latents']
        padding_maps = latent_dict['padding_maps']
        fake_images = self.run_G(runner, style_latents, padding_maps)
        loss_mse = self.mse_loss(fake_images, real_images)
        if self.lpips_weight > 0:
            if self.lpips_net == 'alex':
                loss_lpips = self.lpips(real_images, fake_images)
            elif self.lpips_net == 'vgg':
                loss_lpips = self.lpips(x=real_images,
                                        y=fake_images,
                                        return_tensor='lpips').mean()
        else:
            loss_lpips = 0.0
        if self.idsim_weight > 0:
            loss_id = self.compute_idsim_loss(self.arcface_model,
                                              fake_images,
                                              real_images)
        else:
            loss_id = 0.0
        if self.adv_weight > 0:
            fake_scores = self.run_D(runner, fake_images, sync=sync)
            loss_adv = F.softplus(-fake_scores).mean()
        else:
            loss_adv = 0.0
        if self.wnorm_weight > 0:
            batch_size, dim = style_latents.shape[0], style_latents.shape[1]
            avg_latent = runner.models['encoder'].avg_latent
            avg_latent = avg_latent.repeat(batch_size, dim, 1)
            loss_wnorm = self.compute_wnorm_loss(style_latents, avg_latent)
        else:
            loss_wnorm = 0.0

        total_loss = self.mse_weight * loss_mse
        if self.lpips_weight > 0:
            total_loss += self.lpips_weight * loss_lpips
        if self.idsim_weight > 0:
            total_loss += self.idsim_weight * loss_id
        if self.adv_weight > 0:
            total_loss += self.adv_weight * loss_adv
        if self.wnorm_weight > 0:
            total_loss += self.wnorm_weight * loss_wnorm

        # Update the loss logs.
        runner.running_stats.update({'Loss/MSE': loss_mse})
        runner.running_stats.update({'Loss/LPIPS': loss_lpips})
        runner.running_stats.update({'Loss/IDLOSS': loss_id})
        runner.running_stats.update({'Loss/ADVLOSS': loss_adv})
        runner.running_stats.update({'Loss/E': total_loss})
        runner.running_stats.update({'Loss/WLOSS': loss_wnorm})
        return total_loss.mean()

    def D_loss(self, runner, data, sync=True):
        """Computes loss for discriminator."""
        real_images = data['image']
        real_images.requires_grad = True

        with torch.no_grad():
            latent_dict = self.run_E(runner, real_images, sync=sync)
            style_latents = latent_dict['style_latents']
            padding_maps = latent_dict['padding_maps']
            fake_images = self.run_G(runner, style_latents, padding_maps)

        real_scores = self.run_D(runner, real_images, sync=False)
        fake_scores = self.run_D(runner, fake_images, sync=sync)
        loss_fake = F.softplus(fake_scores).mean()
        loss_real = F.softplus(-real_scores).mean()
        d_loss = loss_fake + loss_real

        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(real_images,
                                                          real_scores)
            grad_penalty = real_grad_penalty * (self.r1_gamma * 0.5)
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(fake_images,
                                                          fake_scores)
            grad_penalty = fake_grad_penalty * (self.r2_gamma * 0.5)

        # calc total loss.
        loss = d_loss + grad_penalty
        # Update the D loss logs.
        runner.running_stats.update({'Loss/D_total': loss.item()})
        runner.running_stats.update({'Loss/D': d_loss.item()})
        runner.running_stats.update({'Loss/GP': grad_penalty.item()})
        return loss
