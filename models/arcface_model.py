"""Contains the ArcFace model, which is used for inference ONLY.

ArcFace model is commonly used to compute face identity similarity, which is
particularly helpful for face-related tasks (both as the training regularizer,
and as the evaluation metric).

This file modifies the implementation from

https://github.com/TreB1eN/InsightFace_Pytorch

which is different from the official implementation from

https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

TODO: Support official implementation, which has the correct version of IResNet.

The training of ArcFace can be found in paper (not included in this file)

https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
"""

import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.misc import download_url

__all__ = ['ArcFaceModel']

# pylint: disable=line-too-long
_MODEL_URL_SHA256 = {
    # This model is provided by https://github.com/TreB1eN/InsightFace_Pytorch
    'ir50_se': (
        'https://gg0ltg.by.files.1drv.com/y4m3fNNszG03z9n8JQ7EhdtQKW8tQVQMFBisPVRgoXi_UfP8pKSSqv8RJNmHy2JampcPmEazo_Mx6NTFSqBpZmhPniROm9uNoghnzaavvYpxkCfiNmDH9YyIF3g-0nwt6bsjk2X80JDdL5z88OAblSDmB-kuQkWSWvA9BM3Xt8DHMCY8lO4HOQCZ5YWUtFyPAVwEyzTGDM-JRA5EJoN2bF1cg',
        'a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28'
    )
}
# pylint: enable=line-too-long


class ArcFaceModel(object):
    """Defines the ArcFace model, which is based on IResNet backbone.

    This is a static class, which is used to avoid this model to be built
    repeatedly. Consequently, this model is particularly used for inference,
    like computing face identity similarity as the training regularizer, or the
    evaluation metric. If training is required, please refer to
    https://github.com/deepinsight/insightface or implement by yourself.

    NOTE: The pre-trained model assumes the inputs to be with `RGB` channel
    order and pixel range [-1, 1], and will resize the input to shape [256, 256]
    automatically.
    """
    models = dict()

    @staticmethod
    def build_model():
        """Builds the model and load pre-trained weights.

        NOTE: Currently, only IResNet-50 backbone with Squeeze-and-Excitation
        (SE) module is supported.
        """
        model_source = 'ir50_se'
        input_size = 112
        num_layers = 50
        use_se = True

        fingerprint = model_source

        if fingerprint not in ArcFaceModel.models:
            # Build model.
            model = ArcFace(input_size=input_size,
                            num_layers=num_layers,
                            use_se=use_se)

            # Download pre-trained weights.
            if dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier()  # Download by chief.

            url, sha256 = _MODEL_URL_SHA256[model_source]
            filename = f'arcface_model_{model_source}_{sha256}.pth'
            model_path = '/input/qingyan/eccv/model_ir_se50.pth'
            if not os.path.exists(model_path):
                model_path, hash_check = download_url(url,
                                                      filename=filename,
                                                      sha256=sha256)
                state_dict = torch.load(model_path, map_location='cpu')
                if hash_check is False:
                    warnings.warn(f'Hash check failed! The remote file from URL '
                                  f'`{url}` may be changed, or the downloading is '
                                  f'interrupted. The loaded arcface model may have '
                                  f'unexpected behavior.')
            else:
                state_dict = torch.load(model_path, map_location='cpu')

            if dist.is_initialized() and dist.get_rank() == 0:
                dist.barrier()  # Wait for other replicas.

            # Load weights.
            model.load_state_dict(state_dict, strict=True)
            del state_dict

            # For inference only.
            model.eval().requires_grad_(False).cuda()
            ArcFaceModel.models[fingerprint] = model

        return ArcFaceModel.models[fingerprint]

# pylint: disable=missing-function-docstring


class ArcFace(nn.Module):
    """Defines the ArcFace model with IResNet as the backbone.

    It is optional to also introduce the Squeeze-and-Excitation (SE) module.

    This model takes `RGB` images with data format `NCHW` as the raw inputs. The
    pixel range are assumed to be [-1, 1]. Input images will be first resized
    to shape [256, 256], centrally cropped, and resized to `input_size`.
    """

    num_stages = 4
    stage_in_channels = [64, 64, 128, 256]
    stage_depth = [64, 128, 256, 512]
    stage_num_blocks = {
        50: [3, 4, 14, 3],  # IResNet-50
        100: [3, 13, 30, 3],  # IResNet-100
        152: [3, 8, 36, 3]  # IResNet-152
    }

    def __init__(self,
                 input_size=112,
                 num_layers=50,
                 use_se=True,
                 se_reduction=16,
                 output_affine=True):
        """Initializes the network with hyper-parameters.

        Args:
            input_size: The input size of the backbone. Support 112 and 224.
                (default: 112)
            num_layers: Number of layers of the backbone. Support 50, 100, and
                152. (default: 50)
            use_se: Whether to use Squeeze-and-Excitation (SE) module as the
                tailing module of the bottleneck block. (default: True)
            se_reduction: Reduction factor of the SE module if used.
                (default: 16)
            output_affine: Whether to learn an affine transformation in the
                final output layer. (default: True)
        """
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.output_affine = output_affine

        assert input_size in [112, 224]
        assert num_layers in self.stage_num_blocks

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        modules = []
        for stage in range(self.num_stages):
            in_channels = self.stage_in_channels[stage]
            depth = self.stage_depth[stage]
            num_blocks = self.stage_num_blocks[self.num_layers][stage]
            for block_idx in range(num_blocks):
                modules.append(IBottleneck(
                    in_channels=in_channels if block_idx == 0 else depth,
                    depth=depth,
                    stride=2 if block_idx == 0 else 1,
                    use_se=use_se,
                    se_reduction=se_reduction))
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * ((input_size // 16) ** 2), 512),
            nn.BatchNorm1d(512, affine=output_affine)
        )

    def forward(self, x):
        # Resize if needed.
        if x.shape[2:4] != (256, 256):
            x = F.interpolate(x,
                              size=(256, 256),
                              mode='bilinear',
                              align_corners=False)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region.
        x = F.adaptive_avg_pool2d(x, (self.input_size, self.input_size))
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x / x.norm(p=2, dim=1, keepdim=True)


class IBottleneck(nn.Module):
    """Defines the bottleneck used in IResNet."""

    def __init__(self, in_channels, depth, stride, use_se, se_reduction):
        """Initializes the bottlenet block.

        Args:
            in_channels: Number of channels of the input tensor.
            depth: Depth of the current block, i.e., output channels.
            stride: Stride of the second convolutional layer, and the shortcut
                layer.
            use_se: Whether to use Squeeze-and-Excitation (SE) module as the
                tailing module of the bottleneck block.
            se_reduction: Reduction factor of the SE module if used.
        """
        super().__init__()

        self.in_channels = in_channels
        self.depth = depth
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction

        if in_channels == depth:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=depth,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=depth)
            )

        modules = [
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=depth,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.PReLU(num_parameters=depth),
            nn.Conv2d(in_channels=depth,
                      out_channels=depth,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=depth),
        ]
        if use_se:
            modules.append(SEModule(depth, se_reduction))
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

# pylint: enable=missing-function-docstring
