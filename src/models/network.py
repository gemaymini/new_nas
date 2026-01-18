# -*- coding: utf-8 -*-
"""
Network building blocks for searched architectures.
"""
import torch
import torch.nn as nn
from typing import List
from configuration.config import config
from core.encoding import Encoder, BlockParams


def get_activation(activation_type: int) -> nn.Module:
    """
    Return the activation module for the given type.

    Args:
        activation_type (int): 0 for ReLU, 1 for SiLU.

    Returns:
        nn.Module: Activation layer.
    """
    if activation_type == 1:
        return nn.SiLU(inplace=False)
    else:
        return nn.ReLU(inplace=False)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of SE block.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvUnit(nn.Module):
    """
    Standard convolution block (Conv-BN-ReLU).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RegBlock(nn.Module):
    """
    Regularized evolution block (MBConv-like structure).
    """
    def __init__(self, in_channels: int, block_params: BlockParams):
        """
        Initialize the block.

        Args:
            in_channels (int): Input channel count.
            block_params (BlockParams): Hyperparameters for this block.
        """
        super(RegBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = block_params.out_channels
        self.pool_stride = block_params.pool_stride
        # Output channels scale with per-block expansion.
        self.out_channels = self.mid_channels * block_params.expansion
        self.groups = block_params.groups
        self.has_senet = block_params.has_senet == 1
        self.activation_type = block_params.activation_type
        self.dropout_rate = block_params.dropout_rate
        self.skip_type = block_params.skip_type  # 0=add, 1=concat, 2=none
        self.kernel_size = block_params.kernel_size

        # Adjust output channels for concat skips.
        if self.skip_type == 1:  # concat
            self.final_out_channels = self.out_channels + in_channels
        else:
            self.final_out_channels = self.out_channels

        self.conv1 = nn.Conv2d(in_channels, self.mid_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)

        padding = self.kernel_size // 2
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels,
                               kernel_size=self.kernel_size, stride=1, padding=padding,
                               groups=self.groups, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)

        if block_params.pool_type == 0:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=block_params.pool_stride, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=block_params.pool_stride, padding=1)

        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        # Shortcut projection as needed by skip type.
        if self.skip_type == 0:  # add
            if block_params.pool_stride != 1 or in_channels != self.out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels,
                              kernel_size=1, stride=block_params.pool_stride, bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )
            else:
                self.shortcut = nn.Identity()
        elif self.skip_type == 1:  # concat
            if block_params.pool_stride != 1:
                self.shortcut = nn.AvgPool2d(kernel_size=3,
                                             stride=block_params.pool_stride,
                                             padding=1)
            else:
                self.shortcut = nn.Identity()
        else:  # none
            self.shortcut = None

        if self.has_senet:
            se_channels = self.final_out_channels
            reduction = config.SENET_REDUCTION
            self.se = SEBlock(se_channels, reduction)
        else:
            self.se = None

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_rate)
        else:
            self.dropout = None

        self.activation = get_activation(self.activation_type)

    def forward(self, x):
        if self.skip_type in (0, 1):
            identity = self.shortcut(x)
        else:
            identity = None

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip_type == 0:  # add
            out = out + identity
        elif self.skip_type == 1:  # concat
            out = torch.cat([out, identity], dim=1)

        if self.se is not None:
            out = self.se(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.activation(out)
        return out

    def get_output_channels(self) -> int:
        return self.final_out_channels


class RegUnit(nn.Module):
    """
    A unit composed of multiple RegBlocks.
    """
    def __init__(self, in_channels: int, block_params_list: List[BlockParams]):
        super(RegUnit, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for block_params in block_params_list:
            block = RegBlock(current_channels, block_params)
            self.blocks.append(block)
            current_channels = block.get_output_channels()

        self.out_channels = current_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_output_channels(self) -> int:
        return self.out_channels


class SearchedNetwork(nn.Module):
    """
    The full neural network constructed from a genetic encoding.
    """
    def __init__(self, encoding: List[int], input_channels: int = 3, num_classes: int = 10):
        super(SearchedNetwork, self).__init__()

        unit_num, _, block_params_list = Encoder.decode(encoding)

        self.conv_unit = ConvUnit(
            in_channels=input_channels,
            out_channels=config.INIT_CONV_OUT_CHANNELS,
            kernel_size=config.INIT_CONV_KERNEL_SIZE,
            stride=config.INIT_CONV_STRIDE,
            padding=config.INIT_CONV_PADDING
        )

        self.units = nn.ModuleList()
        current_channels = config.INIT_CONV_OUT_CHANNELS

        for unit_idx in range(unit_num):
            unit = RegUnit(current_channels, block_params_list[unit_idx])
            self.units.append(unit)
            current_channels = unit.get_output_channels()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_channels, num_classes)

        self.encoding = encoding
        self.final_channels = current_channels

        self._init_weights()

    def _init_weights(self):
        """Kaiming Init for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # SENet Initialization:
        # Initialize SE blocks to be close to Identity (scale ~ 1.0)
        # to avoid signal attenuation which ruins NTK condition number.
        for m in self.modules():
            if isinstance(m, SEBlock):
                # m.fc is Sequential(Linear, ReLU, Linear, Sigmoid)
                # Initialize first Linear
                if isinstance(m.fc[0], nn.Linear):
                    nn.init.kaiming_normal_(m.fc[0].weight, mode='fan_out', nonlinearity='relu')
                
                # Initialize second Linear (output layer)
                if isinstance(m.fc[2], nn.Linear):
                    nn.init.kaiming_normal_(m.fc[2].weight, mode='fan_out', nonlinearity='relu')
                    # Set bias to +5.0 so Sigmoid(x) ~ 1.0
                    if m.fc[2].bias is not None:
                        nn.init.constant_(m.fc[2].bias, 5.0)

        # LowGamma Initialization:
        # Initialize the last BN in each residual block to 0.1.
        # This keeps the block close to identity for stability,
        # but provides enough gradient signal for NTK to obtain non-singular matrices.
        for unit in self.units:
            for block in unit.blocks:
                # RegBlock has bn3 as the last normalization layer
                if hasattr(block, 'bn3') and isinstance(block.bn3, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(block.bn3.weight, 0.1)

    def forward(self, x):
        x = self.conv_unit(x)
        for unit in self.units:
            x = unit(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NetworkBuilder:
    """
    Helper to build SearchedNetwork instances from encodings or individuals.
    """
    @staticmethod
    def build_from_encoding(encoding: List[int], input_channels: int = 3, num_classes: int = 10) -> SearchedNetwork:
        return SearchedNetwork(encoding, input_channels, num_classes)

    @staticmethod
    def build_from_individual(individual, input_channels: int = 3, num_classes: int = 10) -> SearchedNetwork:
        return NetworkBuilder.build_from_encoding(individual.encoding, input_channels, num_classes)

    @staticmethod
    def calculate_param_count(encoding: List[int], input_channels: int = 3, num_classes: int = 10) -> int:
        network = NetworkBuilder.build_from_encoding(encoding, input_channels, num_classes)
        return network.get_param_count()
