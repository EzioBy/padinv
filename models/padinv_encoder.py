# python3.7
"""Contains the implementation of the encoder for PadInv (including pSp).

IResNet is used as the backbone.

PadInv paper: https://arxiv.org/pdf/2203.11105.pdf
pSp paper: https://arxiv.org/pdf/2008.00951.pdf

NOTE: Please use `latent_num` and `num_latents_per_level` to control the
inversion space (i.e., W+ space) used in Padding Space and pSp. In addition, pSp
sets `use_position_latent` and `use_layer_padding_head` as `False` by default.
While PadInv uses these two settings to extract padding coefficients.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['PadInvEncoder']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# pylint: disable=missing-function-docstring

class IBottleneck(nn.Module):
    """Defines the bottleneck used in IResNet."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_se,
                 se_reduction,
                 norm_layer=nn.BatchNorm2d):
        """Initializes the bottleneck block.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of output channels.
            stride: Stride of the second convolutional layer, and the shortcut
                layer.
            use_se: Whether to use Squeeze-and-Excitation (SE) module as the
                tailing module of the bottleneck block.
            se_reduction: Reduction factor of the SE module if used.
            norm_layer: Normalization layer used by this block.
                (default: nn.BatchNorm2d)
        """
        super().__init__()

        self.in_channels = in_channels
        self.depth = out_channels
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction

        if in_channels == out_channels:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                norm_layer(num_features=out_channels)
            )

        modules = [
            norm_layer(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            norm_layer(num_features=out_channels),
        ]
        if use_se:
            modules.append(SEModule(out_channels, se_reduction))
        self.res_layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class SEModule(nn.Module):
    """Defines the Squeeze-and-Excitation (SE) module."""

    def __init__(self, channels, reduction):
        """Initializes the Squeeze-and-Excitation (SE) module.

        Args:
            channels: Number of channels of the input/output tensor.
            reduction: Reduction factor for squeeze.
        """
        super().__init__()

        assert reduction > 0, 'Positive reduction factor is expected!'
        self.channels = channels
        self.reduction = reduction

        self.fc1 = nn.Conv2d(in_channels=channels,
                             out_channels=channels // reduction,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=channels // reduction,
                             out_channels=channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False)

    def forward(self, x):
        module_input = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return module_input * torch.sigmoid(x)


class FPN(nn.Module):
    """Implementation of Feature Pyramid Network (FPN).

    The input of this module is a pyramid of features with reducing resolutions.
    Then, this module fuses these multi-level features from `top_level` to
    `bottom_level`. In particular, starting from the `top_level`, each feature
    is convoluted, upsampled, and fused into its previous feature (which is also
    convoluted).

    Args:
        pyramid_channels: A list of integers, each of which indicates the number
            of channels of the feature from a particular level.
        out_channels: Number of channels for each output.

    Returns:
        A list of feature maps, each of which has `out_channels` channels. The
            elements inside, from left to right, correspond to low-level to
            high-level features.
    """

    def __init__(self, pyramid_channels, out_channels):
        super().__init__()
        assert isinstance(pyramid_channels, (list, tuple))
        self.num_levels = len(pyramid_channels)

        self.lateral_layers = nn.ModuleList()
        for i in range(self.num_levels - 1):
            in_channels = pyramid_channels[i]
            self.lateral_layers.append(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=True))

    def forward(self, inputs):
        if len(inputs) != self.num_levels:
            raise ValueError('Number of inputs and `num_levels` mismatch!')

        # Project all related features to corresponding channels.
        features = []
        for i in range(self.num_levels - 1):
            features.append(self.lateral_layers[i](inputs[i]))
        # The 'top_level' feature needs no operations.
        features.append(inputs[self.num_levels - 1])

        # Fusion, starting from `top_level`.
        for i in range(self.num_levels - 1, 0, -1):
            scale_factor = features[i - 1].shape[2] // features[i].shape[2]
            features[i - 1] = (features[i - 1] +
                               F.interpolate(features[i],
                                             mode='bilinear',
                                             scale_factor=scale_factor,
                                             align_corners=True))
        return features


class EqualLinear(nn.Module):
    """Implements the equal linear layer."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias=True,
                 init_bias=0.0,
                 lr_mul=1):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            lr_mul: Learning multiplier for both weight and bias.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels).div_(lr_mul))
        if add_bias:
            self.bias = nn.Parameter(torch.full([out_channels], init_bias))
        else:
            self.bias = None

        self.scale = (1 / np.sqrt(in_channels)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        out = F.linear(x,
                       weight=self.weight * self.scale,
                       bias=self.bias * self.lr_mul)
        return out

    def extra_repr(self):
        return f'(out_ch={self.weight.shape[1]}, in_ch={self.weight.shape[0]})'


class Map2StyleCodeHead(nn.Module):
    """Implementation of the task-head to produce inverted codes.

    This module was initially proposed in pSp, which utilizes a set of
    convolutions to produce a latent that can be used by a `ModulateConvLayer`.
    """

    def __init__(self, in_channels, out_channels, spatial):
        super().__init__()
        self.out_channels = out_channels
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.LeakyReLU()]
        for _ in range(num_pools - 1):
            modules += [nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1),
                        nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_channels, out_channels, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, self.out_channels)
        x = self.linear(x)
        return x


class PadInvEncoder(nn.Module):
    """Defines the IResNet-based encoder network for GAN inversion.

    On top of the backbone, there are several task-heads to produce inverted
    codes. Please use `latent_dim` and `num_latents_per_level` to define the
    structure. For example, `latent_dim = [512] * 14` and
    `num_latents_per_level = [4, 4, 6]` can be used for StyleGAN inversion with
    14-layer latent codes, where 3 levels (corresponding to 4, 4, 6 layers,
    respectively) are used.

    Settings for the encoder network:

    (1) resolution: The resolution of image to be fed into the encoder.
    (2) latent_dim: Dimension of the latent space. A number (one code will be
        produced), or a list of numbers regarding layer-wise latent codes
        (e.g., passing [512] * 18).
    (3) num_latents_per_level: Number of latents that is produced in each level.
    (4) image_channels: Number of channels of the output image. (default: 3)
    (5) final_res: Final resolution of the convolutional layers. (default: 4)
    (6) start_from_latent_avg: Whether to calculate the W+ latent code based on
        the average latent code. The encoder predicts the residual if this
        is set to `True`. (default: True)

    IResNet-related settings:

    (1) network_depth: Depth of the network. Support 50, 100, and 152.
        (default: 50)
    (2) inplanes: Number of channels of the first convolutional layer.
        (default: 64)
    (3) use_se: Whether to use Squeeze-and-Excitation (SE) module as the tailing
        module of the bottleneck block. (default: True)
    (4) se_reduction: Reduction factor of the SE module, this field takes effect
        if and only if `use_se` is set to `True`. (default: 16)
    (5) norm_layer: Normalization layer used in the encoder. If set as `None`,
        `nn.BatchNorm2d` will be used. Also, please NOTE that when using batch
        normalization, the batch size is required to be larger than one for
        training. (default: nn.BatchNorm2d)

    Task-head related settings:

    (1) fpn_channels: Number of channels used in Feature Pyramid Network (FPN).
        (default: 512)

    PadInv related Settings:

    (1) use_padding_space: Whether to use padding coefficients extracted by
        encoder to pad the convolution layers. (default: True)
    (2) encode_const_input: Whether to replace the constant input with the
        encoder output. (default: True)
    (3) only_pad_even: Whether to only pad even convolution layers, i.e., the
        convolution layers with `kernel_size` greater than 1 and `scale_factor`
        equals to 1. This field only takes effect when `use_padding_space` is
        set to `True`. (default: True)
    (4) pad_layer_num: How many convolution layers will be padded with padding
        coefficients. Note that 0 indicates only constant input will be replaced
        with the padding coefficients. This field only takes effect when
        `use_padding_space` is set to `True`. Currently the supported max number
        of padding layers is 5. (default: 4)
    (5) depth_padding_extractor: Amount of IResNet bottlenecks to produce
        padding maps. This field only takes effect when `use_padding_space` is
        set to `True`. (default: 2)
    (6) use_padding_modulation_head: Whether to use several 1x1 convolutions
        to modulate the generated padding latents FOR EACH LAYER. This field
        only takes effect when `use_padding_space` is set to `True`.
        (default: True)
    """

    # The following settings are consistent across IResNet-50/100/152.
    stage_in_channels = [64, 64, 128, 256]  # `in_channels` in each stage
    stage_out_channels = [64, 128, 256, 512]  # `out_channels` in each stage

    # Number of blocks is specified by backbone architecture.
    arch_settings = {
        50: (IBottleneck, [3, 4, 14, 3]),  # IResNet-50
        100: (IBottleneck, [3, 13, 30, 3]),  # IResNet-100
        152: (IBottleneck, [3, 8, 36, 3])  # IResNet-152
    }

    def __init__(self,
                 # Settings for encoder network.
                 resolution,
                 latent_dim,
                 num_latents_per_level,
                 image_channels=3,
                 final_res=4,
                 start_from_latent_avg=True,
                 # Settings for IResNet backbone.
                 network_depth=50,
                 inplanes=64,
                 use_se=True,
                 se_reduction=16,
                 norm_layer=nn.BatchNorm2d,
                 # Settings for task head.
                 fpn_channels=512,
                 # Settings for PadInv.
                 use_padding_space=True,
                 encode_const_input=True,
                 only_pad_even=True,
                 pad_layer_num=4,
                 depth_padding_extractor=2,
                 use_padding_modulation_head=True):
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if network_depth not in self.arch_settings:
            raise ValueError(f'Invalid network depth: `{network_depth}`!\n'
                             f'Options allowed: '
                             f'{list(self.arch_settings.keys())}.')
        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        assert isinstance(latent_dim, (list, tuple))
        assert isinstance(num_latents_per_level, (list, tuple))
        assert sum(num_latents_per_level) == len(latent_dim)

        self.resolution = resolution
        self.latent_dim = latent_dim
        self.num_latents_per_level = num_latents_per_level
        self.num_fpn_levels = len(self.num_latents_per_level)
        self.image_channels = image_channels
        self.final_res = final_res
        self.inplanes = inplanes
        self.network_depth = network_depth
        if norm_layer is None or norm_layer == 'nn.BatchNorm2d':
            norm_layer = nn.BatchNorm2d
        if norm_layer == nn.BatchNorm2d and dist.is_initialized():
            norm_layer = nn.SyncBatchNorm
        self.norm_layer = norm_layer
        self.fpn_channels = fpn_channels
        self.start_from_latent_avg = start_from_latent_avg
        self.avg_latent = None
        self.use_padding_space = use_padding_space
        self.encode_const_input = encode_const_input
        self.only_pad_even = only_pad_even
        self.pad_layer_num = pad_layer_num
        if not 0 < self.pad_layer_num <= 5:
            raise ValueError('The maximum number of padding layer is set to 5 '
                             '(equivalent to pad 64x64 feature map at most) '
                             'considering the trade of between accuracy and '
                             'efficiency.')
        self.depth_padding_extractor = depth_padding_extractor
        self.use_padding_modulation_head = use_padding_modulation_head

        block_fn, num_blocks_per_stage = self.arch_settings[network_depth]

        self.num_stages = len(num_blocks_per_stage)
        assert self.num_stages > 1

        # Normally encoder starts with a 3x3 convolution.
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.image_channels,
                      out_channels=self.inplanes,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            norm_layer(inplanes),
            nn.PReLU(num_parameters=inplanes))

        # Architecture of IR or IR-SE based encoder.
        self.backbone_stages = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            blocks_this_stage = []
            in_channels = self.stage_in_channels[stage_idx]
            out_channels = self.stage_out_channels[stage_idx]
            num_blocks = num_blocks_per_stage[stage_idx]
            for block_idx in range(num_blocks):
                blocks_this_stage.append(block_fn(
                    in_channels=in_channels if block_idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=2 if block_idx == 0 else 1,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    norm_layer=norm_layer))
            self.backbone_stages.append(nn.Sequential(*blocks_this_stage))

        # Modules to generate padding coefficients for padding space.
        if self.use_padding_space:  # PadInv Encoder
            padding_extracting_modules = []
            for block_idx in range(self.depth_padding_extractor):
                padding_extracting_modules.append(
                    block_fn(in_channels=fpn_channels,
                             out_channels=fpn_channels,
                             stride=1,
                             use_se=use_se,
                             se_reduction=se_reduction,
                             norm_layer=norm_layer))

            self.padding_extractor = nn.Sequential(*padding_extracting_modules)

            if self.use_padding_modulation_head:
                # The 1x1 convolutions for padding modulation.
                self.padding_modulation_heads = nn.ModuleList()
                for _ in range(self.pad_layer_num):
                    self.padding_modulation_heads.append(
                        nn.Conv2d(in_channels=fpn_channels,
                                  out_channels=fpn_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
                    )

        if self.num_fpn_levels > len(self.stage_out_channels):
            raise ValueError('Number of task heads is larger than number of '
                             'stages! Please reduce the number of heads.')

        # FPN.
        fpn_pyramid_channels = self.stage_out_channels[-self.num_fpn_levels:]
        self.fpn = FPN(pyramid_channels=fpn_pyramid_channels,
                       out_channels=self.fpn_channels)
        assert self.num_fpn_levels == self.fpn.num_levels

        # Initialize code heads.
        self.latent_heads = nn.ModuleList()
        head_idx = 0
        for level_idx in range(self.num_fpn_levels):
            num_heads = self.num_latents_per_level[level_idx]
            for _ in range(num_heads):
                self.latent_heads.append(
                    Map2StyleCodeHead(in_channels=fpn_channels,
                                      out_channels=self.latent_dim[head_idx],
                                      spatial=16 * (2 ** level_idx))
                )
                head_idx += 1

    def forward(self, x, feature_idx=1):
        """Gets encoded style latents (and padding maps if use PadInv).

        NOTE: `feature_idx` only takes effect when `self.use_padding_space` is
        set to `True`. In such a setting, the primal padding map is derived from
        the feature selected by `feature_idx` within the outputs of Feature
        Pyramid Network (FPN). The default value `1` is an empirical choice.
        """
        x = self.input_layer(x)

        # Forwarding the encoder backbone.
        features = []
        # Obtain the hierarchical feature maps.
        for i in range(self.num_stages):
            x = self.backbone_stages[i](x)
            features.append(x)
        features = features[-self.num_fpn_levels:]

        # Post-process features with Feature Pyramid Network (FPN).
        features = self.fpn(features)
        # Reorder features to high-level (low-resolution) first.
        features = features[::-1]

        # Obtain style latent codes (in W+ space) from code heads.
        style_latents = []
        head_idx = 0
        for level_idx in range(self.num_fpn_levels):
            num_heads = self.num_latents_per_level[level_idx]
            for _ in range(num_heads):
                code = self.latent_heads[head_idx](features[level_idx])
                style_latents.append(code)
                head_idx += 1
        style_latents = torch.stack(style_latents, dim=1)

        # Learn the style latent code originated from the average.
        if self.start_from_latent_avg:
            batch_size = style_latents.shape[0]
            num_styles = style_latents.shape[1]
            style_latents += self.avg_latent.repeat(batch_size, num_styles, 1)

        # Prepare the result dict.
        # 'style_latents' is a tensor shaped as [batch_size, num_layers, 512].
        empty_padding_maps = [None] * len(self.latent_heads)
        latent_dict = dict(style_latents=style_latents,
                           padding_maps=empty_padding_maps)

        if not self.use_padding_space:  # pixel2style2pixel
            return latent_dict

        # Obtain the padding coefficient map,
        # currently shaped as [batch_size, 512, 32, 32] obtained by convolving
        # the 32x32 FPN feature map with the padding extractor (and modulation
        # head).
        padding_map = self.padding_extractor(features[feature_idx])
        target_padding_size = 2 ** (self.pad_layer_num + 1) + 2

        # Upsampling to get the feature map with higher resolution
        # corresponding to the required padding size.
        padding_map = F.interpolate(padding_map,
                                    mode='bicubic',
                                    size=(target_padding_size,
                                          target_padding_size),
                                    align_corners=True)

        initial_input = padding_map if self.encode_const_input else None
        latent_dict['padding_maps'][0] = initial_input

        if not self.use_padding_modulation_head:
            # Use the same padding map as padding coefficient for each layer.
            for i in range(self.pad_layer_num):
                idx = i * 2 if self.only_pad_even else i
                idx += 1  # Reserve 0-index for initial_input.
                latent_dict['padding_maps'][idx] = padding_map
            return latent_dict

        # Learn padding coefficients for each layer.
        padding_map_list = []
        for padding_modulation_head in self.padding_modulation_heads:
            modulated_padding = padding_modulation_head(padding_map)
            padding_map_list.append(modulated_padding)
        if self.only_pad_even:
            for idx, layer_padding in enumerate(padding_map_list):
                latent_dict['padding_maps'][2 * idx + 1] = layer_padding
        else:
            num_maps = len(padding_map_list)
            latent_dict['padding_maps'][:num_maps] = padding_map_list
        return latent_dict

# pylint: enable=missing-function-docstring


if __name__ == '__main__':
    encoder = PadInvEncoder(resolution=256,
                            latent_dim=[512] * 18,
                            start_from_latent_avg=False,  # For test only.
                            num_latents_per_level=[3, 4, 11])
    input_img = torch.randn((2, 3, 256, 256))
    # Return the W+ latent code list shaped as [512] * 18.
    latent_code = encoder(input_img)
