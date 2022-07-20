# python3.7
"""Configuration for GAN Inversion by training PadInv encoder."""
import click

from .base_config import BaseConfig

__all__ = ['PadInvEncoderConfig']

RUNNER = 'PadInvRunner'
DATASET = 'ImageDataset'
ENCODER = 'PadInvEncoder'
GENERATOR = 'StyleGAN2GeneratorPadInv'
DISCRIMINATOR = 'StyleGAN2Discriminator'
LOSS = 'GANInversionLoss'


class PadInvEncoderConfig(BaseConfig):
    """Defines the configuration for PadInv encoder."""

    name = 'padinv_encoder'
    hint = 'Train a PadInv Encoder for StyleGAN2.'
    info = '''
To train an encoder for StyleGAN2 model, the recommended settings are as 
follows:

\b
- use_padding_space: True (for PadInv Encoder)
- batch_size: 8 (for FF-HQ dataset)
- val_batch_size: 8 (for FF-HQ dataset)
- total_img: 6_400_000 (for FF-HQ dataset)
- train_data_mirror: True (for FF-HQ dataset)
'''

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.config.runner_type = RUNNER

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Data transformation settings'].extend([
            cls.command_option(
                '--encode_resolution', type=cls.int_type, default=256,
                help='Resolution to fed into encoder.'),
            cls.command_option(
                '--gen_resolution', type=cls.int_type, default=256,
                help='Resolution of the generated image.'),
            cls.command_option(
                '--image_channels', type=cls.int_type, default=3,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val', type=cls.float_type, default=-1.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val', type=cls.float_type, default=1.0,
                help='Maximum pixel value of the training images.')
        ])

        options['Encoder network settings'].extend([
            cls.command_option(
                '--latent_dim', type=cls.json_type, default=18 * [512],
                help='Dimension of the latent space. A number (one code will '
                     'be produced) or a list of numbers regarding layer-wise '
                     'latent codes.'),
            cls.command_option(
                '--num_latents_per_level', type=cls.json_type,
                default=[3, 4, 11],
                help='Number of latents that is produced in each level.'),
            cls.command_option(
                '--start_from_latent_avg', type=cls.bool_type, default=True,
                help='Whether to calculate the W+ latent code based on the '
                     'average latent code. The encoder predicts the residual '
                     'if this setting to `True`.'),
            cls.command_option(
                '--final_res', type=cls.int_type, default=4,
                help='Final resolution of the convolutional layers.'),
            cls.command_option(
                '--network_depth', type=click.Choice(['50', '100', '152']),
                default='50',
                help='Depth of the IResNet backbone.'),
            cls.command_option(
                '--inplanes', type=cls.int_type, default=64,
                help='Number of channels of the first convolutional layer.'),
            cls.command_option(
                '--use_se', type=cls.bool_type, default=True,
                help='Whether to use Squeeze-and-Excitation (SE) module as the '
                     'tailing module of the bottleneck block.'),
            cls.command_option(
                '--se_reduction', type=cls.int_type, default=16,
                help='Reduction factor of the SE module, this field takes '
                     'effect if and only if `use_se` is set to `True`.'),
            cls.command_option(
                '--norm_layer', type=str, default='nn.BatchNorm2d',
                help='Normalization layer used in the encoder. If set as '
                     '`None`, `nn.BatchNorm2d` will be used. Also, please NOTE '
                     'that when using batch normalization, the batch size is '
                     'required to be larger than one for training.'),
            cls.command_option(
                '--fpn_channels', type=cls.int_type, default=512,
                help='Number of channels used in Feature Pyramid Network (FPN).'
            ),
            cls.command_option(
                '--use_padding_space', type=cls.bool_type, default=True,
                help='Whether to use padding coefficients extracted by encoder '
                     'to pad the convolution layers.'),
            cls.command_option(
                '--encode_const_input', type=cls.bool_type, default=True,
                help='Whether to use padding coefficients extracted by encoder '
                     'to replace the original constant input parameters.'),
            cls.command_option(
                '--only_pad_even', type=cls.bool_type, default=True,
                help='Whether to only pad even convolution layers, i.e., the '
                     'convolution layers with `kernel_size` greater than 1 and '
                     '`scale_factor` equals to 1. This field only takes effect '
                     'when `use_padding_space` is set to `True`.'),
            cls.command_option(
                '--pad_layer_num', type=cls.int_type, default=4,
                help='How many convolution layers will be padded with padding '
                     'coefficients. Note that 0 indicates only constant input '
                     'will be replaced with the padding coefficients. This '
                     'field only takes effect when `use_padding_space` is set '
                     'to `True`.'),
            cls.command_option(
                '--depth_padding_extractor', type=cls.int_type, default=2,
                help='Amount of IResNet bottlenecks to produce padding maps. '
                     'This field only takes effect when `use_padding_space` is '
                     'set to `True`.'),
            cls.command_option(
                '--use_pad_mod_head', type=cls.bool_type,
                default=False,
                help='Whether to use several 1x1 convolutions to modulate the '
                     'generated padding latents FOR EACH LAYER. This field '
                     'only takes effect when `use_padding_space` is set to '
                     '`True`.')
        ])

        options['GAN network settings'].extend([
            cls.command_option(
                '--g_init_res', type=cls.int_type, default=4,
                help='The initial resolution to start convolution with in '
                     'generator.'),
            cls.command_option(
                '--z_dim', type=cls.int_type, default=512,
                help='The dimension of the latent space.'),
            cls.command_option(
                '--d_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'discriminator, which will be `factor * 32768`.'),
            cls.command_option(
                '--g_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * 32768`.'),
            cls.command_option(
                '--g_num_mappings', type=cls.int_type, default=8,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--g_architecture', type=str, default='skip',
                help='Architecture type of generator.'),
            cls.command_option(
                '--d_architecture', type=str, default='resnet',
                help='Architecture type of discriminator.'),
            cls.command_option(
                '--impl', type=str, default='cuda',
                help='Control the implementation of some neural operations.'),
            cls.command_option(
                '--num_fp16_res', type=cls.int_type, default=0,
                help='Number of (highest) resolutions that use `float16` '
                     'precision for training, which speeds up the training yet '
                     'barely affects the performance. The official '
                     'StyleGAN-ADA uses 4 by default.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--pretrained_gan_path', type=str,
                help='Path to a pretrained StyleGAN2 checkpoint.'),
            cls.command_option(
                '--e_lr', type=cls.float_type, default=1e-4,
                help='The learning rate of encoder.'),
            cls.command_option(
                '--e_beta_1', type=cls.float_type, default=0.9,
                help='The Adam hyper-parameter `beta_1` for encoder '
                     'optimizer.'),
            cls.command_option(
                '--e_beta_2', type=cls.float_type, default=0.999,
                help='The Adam hyper-parameter `beta_2` for encoder '
                     'optimizer.'),
            cls.command_option(
                '--mse_weight', type=cls.float_type, default=1.0,
                help='Factor to control the strength of Mean Square Error '
                     '(MSE) regularization.'),
            cls.command_option(
                '--lpips_net', type=str, default='alex',
                help='The network architecture of LPIPS.'),
            cls.command_option(
                '--lpips_weight', type=cls.float_type, default=0.8,
                help='Factor to control the strength of Learned Perceptual '
                     'Image Patch Similarity (LPIPS) regularization.'),
            cls.command_option(
                '--idsim_weight', type=cls.float_type, default=0.1,
                help='Factor to control the strength of identity similarity '
                     '(IDSIM) regularization.'),
            cls.command_option(
                '--wnorm_weight', type=cls.float_type, default=0.0,
                help='Factor to control the strength of W+ Norm '
                     'regularization.'),
            cls.command_option(
                '--adv_weight', type=cls.float_type, default=0.1,
                help='Factor to control the strength of adversarial '
                     'regularization. This field only takes effect when '
                     '`use_discriminator` is set to `True`.'),
            cls.command_option(
                '--use_discriminator', type=cls.bool_type, default=False,
                help='Whether to use the discriminator to assist the training '
                     'of encoder'),
            cls.command_option(
                '--d_mbstd_groups', type=cls.int_type, default=4,
                help='Number of groups for MiniBatchSTD layer of '
                     'discriminator. This field only takes effect when '
                     '`use_discriminator` is set to `True`.'),
            cls.command_option(
                '--d_lr', type=cls.float_type, default=1e-4,
                help='The learning rate of discriminator.'),
            cls.command_option(
                '--d_beta_1', type=cls.float_type, default=0.9,
                help='The Adam hyper-parameter `beta_1` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--d_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--r1_gamma', type=cls.float_type, default=10.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--calc_idsim', type=cls.bool_type, default=False,
                help='Whether to compute Identity similarity (IDSIM) during '
                     'validation.')
        ])
        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'encode_resolution', 'gen_resolution', 'latent_dim',
            'num_latents_per_level', 'network_depth', 'use_se', 'se_reduction',
            'use_padding_space', 'encode_const_input', 'pad_layer_num',
            'depth_padding_extractor', 'use_pad_mod_head', 'z_dim',
            'pretrained_gan_path', 'e_lr', 'mse_weight', 'lpips_net',
            'lpips_weight', 'idsim_weight', 'wnorm_weight', 'adv_weight'
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()

        encode_resolution = self.args.pop('encode_resolution')
        gen_resolution = self.args.pop('gen_resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=encode_resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val
        )
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs
        self.config.data.pretrained_gan_path = self.add_prefetch_file(
            self.args.pop('pretrained_gan_path'))

        g_init_res = self.args.pop('g_init_res')
        d_init_res = 4  # This should be fixed as 4.
        z_dim = self.args.pop('z_dim')
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (32 << 10))
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

        # Parse network settings and training settings.
        if not isinstance(num_fp16_res, int) or num_fp16_res <= 0:
            d_fp16_res = None
            conv_clamp = None
        else:
            d_fp16_res = max(gen_resolution // (2 ** (num_fp16_res - 1)),
                             d_init_res * 2)
            conv_clamp = 256

        self.config.encode_resolution = encode_resolution
        e_beta_1 = self.args.pop('e_beta_1')
        e_beta_2 = self.args.pop('e_beta_2')

        num_latents_per_level = self.args.pop('num_latents_per_level')
        start_from_latent_avg = self.args.pop('start_from_latent_avg')

        encode_const_input = self.args.pop('encode_const_input')
        depth_padding_extractor = self.args.pop('depth_padding_extractor')
        self.config.use_discriminator = self.args.pop('use_discriminator')
        use_pad_mod_head = self.args.pop('use_pad_mod_head')

        self.config.models.update(
            encoder=dict(
                model=dict(model_type=ENCODER,
                           resolution=encode_resolution,
                           latent_dim=self.args.pop('latent_dim'),
                           num_latents_per_level=num_latents_per_level,
                           image_channels=image_channels,
                           final_res=self.args.pop('final_res'),
                           start_from_latent_avg=start_from_latent_avg,
                           network_depth=int(self.args.pop('network_depth')),
                           inplanes=self.args.pop('inplanes'),
                           use_se=self.args.pop('use_se'),
                           se_reduction=self.args.pop('se_reduction'),
                           norm_layer=self.args.pop('norm_layer'),
                           fpn_channels=self.args.pop('fpn_channels'),
                           use_padding_space=self.args.pop('use_padding_space'),
                           encode_const_input=encode_const_input,
                           only_pad_even=self.args.pop('only_pad_even'),
                           pad_layer_num=self.args.pop('pad_layer_num'),
                           depth_padding_extractor=depth_padding_extractor,
                           use_padding_modulation_head=use_pad_mod_head),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=self.args.pop('e_lr'),
                         betas=(e_beta_1, e_beta_2))
            ),
            generator=dict(
                model=dict(model_type=GENERATOR,
                           resolution=gen_resolution,
                           image_channels=image_channels,
                           init_res=g_init_res,
                           z_dim=z_dim,
                           mapping_layers=self.args.pop('g_num_mappings'),
                           architecture=self.args.pop('g_architecture'),
                           fmaps_base=g_fmaps_base,
                           conv_clamp=conv_clamp),
                kwargs_val=dict(noise_mode='const',
                                fused_modulate=False,
                                fp16_res=None,
                                impl=impl)
            )
        )

        self.config.loss.update(
            loss_type=LOSS,
            e_loss_kwargs=dict(mse_weight=self.args.pop('mse_weight'),
                               lpips_net=self.args.pop('lpips_net'),
                               lpips_weight=self.args.pop('lpips_weight'),
                               idsim_weight=self.args.pop('idsim_weight'),
                               wnorm_weight=self.args.pop('wnorm_weight'),
                               adv_weight=self.args.pop('adv_weight'))
        )

        self.config.metrics.update(
            GANInversionMetric=dict(
                init_kwargs=dict(name='gan_inversion',
                                 calc_idsim=self.args.pop('calc_idsim')),
                eval_kwargs=dict(
                    generator=dict(),
                    encoder=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=True
            )
        )

        d_architecture = self.args.pop('d_architecture')
        d_mbstd_groups = self.args.pop('d_mbstd_groups')
        d_fmaps_factor = self.args.pop('d_fmaps_factor')
        r1_gamma = self.args.pop('r1_gamma')
        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        r1_interval = self.args.pop('r1_interval')
        if self.config.use_discriminator:
            d_fmaps_base = int(d_fmaps_factor * (32 << 10))

            if r1_interval is not None and r1_interval > 0:
                d_mb_ratio = r1_interval / (r1_interval + 1)
                d_lr = d_lr * d_mb_ratio
                d_beta_1 = d_beta_1 ** d_mb_ratio
                d_beta_2 = d_beta_2 ** d_mb_ratio

            # Append discriminator into models.
            self.config.models.update(
                discriminator=dict(
                    model=dict(model_type=DISCRIMINATOR,
                               resolution=gen_resolution,
                               init_res=d_init_res,
                               architecture=d_architecture,
                               fmaps_base=d_fmaps_base,
                               conv_clamp=conv_clamp,
                               mbstd_groups=d_mbstd_groups),
                    lr=dict(lr_type='FIXED'),
                    opt=dict(opt_type='Adam',
                             base_lr=d_lr,
                             betas=(d_beta_1, d_beta_2)),
                    kwargs_train=dict(fp16_res=d_fp16_res, impl=impl),
                    kwargs_val=dict(fp16_res=None, impl=impl)
                )
            )
            # Take adversarial loss into account.
            self.config.loss.update(
                d_loss_kwargs=dict(r1_gamma=r1_gamma,
                                   r1_interval=r1_interval)
            )
