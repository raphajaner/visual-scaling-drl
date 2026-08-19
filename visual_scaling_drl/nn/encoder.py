"""Neural network visual encoders for deep reinforcement learning (Impala, Impoola, NatureCNN)."""

import numpy as np
import torch
import torch.nn as nn

from visual_scaling_drl.nn.utils import activation_factory


# Taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    """Standard 2-conv residual block for Impala-style architectures."""

    def __init__(self, channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.activation0 = activation_factory(activation)
        self.activation1 = activation_factory(activation)

    def forward(self, x):
        inputs = x
        x = self.activation0(x)
        x = self.conv0(x)
        x = self.activation1(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size=3, activation='relu', strong_downsample=False):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # Input convolution and pooling
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=kernel_size if not strong_downsample else 4,
                              padding="same")
        if strong_downsample:
            self.pooling = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block0 = ResidualBlock(self._out_channels, kernel_size=kernel_size, activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, kernel_size=kernel_size, activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpoolaCNN(nn.Module):
    """
    Impoola CNN encoder augmented with Global Average Pooling (GAP) for resolution-independent visual DRL.

    The Impoola architecture introduces a Global Average Pooling layer prior to linear projection heads,
    decoupling the network's trainable parameter count from the spatial dimensions (H, W) of input observations.
    This enables visual scaling to higher resolutions (e.g. 96x96, 112x112) without incurring quadratic
    parameter growth or spatial overfitting.

    Args:
        envs: Vectorized environment instance providing observation/action space specifications.
        width_scale (int): Channel width scaling factor multiplier (tau). Default: 1.
        out_features (int): Dimensionality of latent feature output projection. Default: 256.
        cnn_filters (tuple[int, ...]): Output channels for each ConvSequence block. Default: (16, 32, 32).
        kernel_size (int): Convolutional kernel spatial dimension. Default: 3.
        activation (str): Nonlinear activation function name. Default: 'relu'.
        pooling_type (str): Pooling layer mode ('avg' for Global Average Pooling, None for flattening). Default: 'avg'.
        pooling_layer_kernel_size (int): Output spatial grid size for adaptive pooling. Default: 1.

    Reference:
        "Impoola: The Power of Average Pooling for Image-based Deep Reinforcement Learning"
        Raphael Trumpp, Ansgar Schäfftlein, Mirco Theile, Marco Caccamo.
        Reinforcement Learning Conference (RLC) / Reinforcement Learning Journal (RLJ), 2025.
        https://openreview.net/forum?id=Kkw4nqaM9Y
    """

    def __init__(
            self, envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), kernel_size=3, activation='relu',
            pooling_type="avg", pooling_layer_kernel_size=1,
            use_dropout=False,
            use_drqv2_style=False,
            use_relu_after_last_conv=True,
            strong_downsample_first=False,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.pooling_layer_kernel_size = pooling_layer_kernel_size
        self.use_relu_after_last_conv = use_relu_after_last_conv
        self.use_drqv2_style = use_drqv2_style

        shape = envs.single_observation_space.shape  # (c, h, w)

        # CNN backbone
        cnn_layers = []
        for i_block, out_channels in enumerate(cnn_filters):
            conv_seq = ConvSequence(
                shape, int(out_channels * width_scale), kernel_size=kernel_size,
                activation=activation,
                strong_downsample=(i_block == 0) if strong_downsample_first else False
            )
            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)

        # Final activation after last ConvSequence, not included in the ConvSequence itself
        if self.use_relu_after_last_conv:
            cnn_layers += [activation_factory(activation)]

        # Pooling layers after conv blocks, before Linear projection head.
        if self.pooling_type is None:
            pass
        elif self.pooling_type == "avg":
            cnn_layers += [nn.AdaptiveAvgPool2d((pooling_layer_kernel_size, pooling_layer_kernel_size))]
        elif self.pooling_type == "max":
            cnn_layers += [nn.AdaptiveMaxPool2d((pooling_layer_kernel_size, pooling_layer_kernel_size))]
        elif self.pooling_type == "1d_conv":
            cnn_layers += [nn.Conv2d(
                in_channels=shape[0], out_channels=shape[0], kernel_size=(1, shape[1]), padding=0)]
        elif self.pooling_type == "depthwise_global_conv":
            cnn_layers += [nn.Conv2d(
                in_channels=shape[0], out_channels=shape[0], kernel_size=shape[1], groups=shape[0], padding=0)]
        elif self.pooling_type == "depthwise_conv":
            cnn_layers += [nn.Conv2d(
                in_channels=shape[0], out_channels=shape[0], kernel_size=3, padding=1, groups=shape[0])]
            raise NotImplementedError

        # Linear head
        linear_layers = cnn_layers
        linear_layers += [nn.Flatten()]

        if use_dropout:
            linear_layers += [nn.Dropout(0.1)]

        if self.use_drqv2_style:
            linear_layers += [
                nn.LazyLinear(out_features),
                nn.LayerNorm(out_features),
                nn.Tanh()
            ]
        else:
            linear_layers += [
                nn.LazyLinear(out_features),
                activation_factory(activation)
            ]

        self.network = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x / 255.0
        return self.network(x)


class ImpalaCNN(ImpoolaCNN):
    """
    Standard Impala-CNN encoder architecture for deep reinforcement learning (Espeholt et al., 2018).

    Impala-CNN consists of three convolutional residual blocks that downsample spatial feature maps
    via 3x3 max-pooling layers. It flattens spatial feature maps directly into fully connected layers
    without a global pooling layer, resulting in O(H * W) parameter scaling as input resolution increases.

    Args:
        envs: Vectorized environment instance providing observation/action space specifications.
        width_scale (int): Channel width scaling factor multiplier (tau). Default: 1.
        out_features (int): Dimensionality of latent feature output projection. Default: 256.
        cnn_filters (tuple[int, ...]): Output channels for each ConvSequence block. Default: (16, 32, 32).
        kernel_size (int): Convolutional kernel spatial dimension. Default: 3.
        activation (str): Nonlinear activation function name. Default: 'relu'.

    Reference:
        "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
        Lasse Espeholt et al., ICML 2018.
    """

    def __init__(self, envs,
                 width_scale=1, out_features=256, cnn_filters=(16, 32, 32), kernel_size=3, activation='relu',
                 use_dropout=False,
                 use_relu_after_last_conv=True,
                 use_drqv2_style=False,
                 strong_downsample_first=False,
                 ):
        if not use_relu_after_last_conv:
            raise ValueError("ImpalaCNN only supports ReLU after last convolutional layer")

        super().__init__(envs,
                         width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters,
                         kernel_size=kernel_size, activation=activation,
                         pooling_type=None,  # None means flatten after the last ConvSequence without any pooling
                         pooling_layer_kernel_size=1,
                         use_dropout=use_dropout,
                         use_relu_after_last_conv=True,
                         strong_downsample_first=strong_downsample_first,
                         use_drqv2_style=use_drqv2_style
                         )


class NatureCNN(nn.Module):
    def __init__(self, envs,
                 width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu', pooling_type=None,
                 *args, **kwargs,
                 ):
        super().__init__()

        shape = envs.single_observation_space.shape  # (c, h, w)

        layers = [
            nn.Conv2d(shape[0], cnn_filters[0] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[0] * width_scale, cnn_filters[1] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[1] * width_scale, cnn_filters[1] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[2] * width_scale, cnn_filters[2] * width_scale, 3, stride=1),
            activation_factory(activation)
        ]

        if pooling_type is None:
            pass
        elif pooling_type == "avg":
            layers += [nn.AdaptiveAvgPool2d((1, 1))]
        else:
            raise NotImplementedError

        layers += [
            nn.Flatten(),
            nn.LazyLinear(out_features),  # Linear projection layer
            activation_factory(activation)
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x / 255.0)


def encoder_factory(encoder_type='impoola', use_moe=False, *args, **kwargs):
    if encoder_type == 'impala':
        model = ImpalaCNN(*args, **kwargs)
        if use_moe:
            # remove the linear head and only keep the CNN backbone
            backbone = model.network[:-4]
            assert backbone[-1].__class__ == ConvSequence

            h = w = int(kwargs['envs'].single_observation_space.shape[-1] / 8)
            model = SoftMoE(
                module=ExpertModel,
                backbone=backbone,
                num_experts=10, num_tokens=h * w,
                token_length=kwargs['width_scale'] * kwargs['cnn_filters'][-1],
                expert_hidden_size=kwargs['out_features'],
                capacity_factor=1, expert_type="SMALL", normalization=False, use_random_phi=False
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *kwargs['envs'].single_observation_space.shape)
                out_features = model(dummy).shape[-1]
        else:
            out_features = kwargs['out_features']
        return model, out_features

    elif encoder_type == 'impoola':
        return ImpoolaCNN(*args, **kwargs), kwargs['out_features']

    elif encoder_type == 'nature':
        return NatureCNN(*args, **kwargs), kwargs['out_features']

    else:
        raise NotImplementedError(f"Unsupported encoder type: {encoder_type}")
