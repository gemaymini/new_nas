# -*- coding: utf-8 -*-
"""
网络构建模块
实现conv_unit、reg_block、reg_unit、SENet等网络组件
"""
import torch
import torch.nn as nn
from typing import List
from configuration.config import config
from core.encoding import Encoder, BlockParams
from utils.logger import logger


def get_activation(activation_type: int) -> nn.Module:
    """根据激活函数类型返回对应的激活函数模块"""
    if activation_type == 0:
        return nn.ReLU(inplace=True)
    elif activation_type == 1:
        return nn.SiLU(inplace=True)
    elif activation_type == 2:
        return nn.GELU()
    else:
        return nn.ReLU(inplace=True)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RegBlock(nn.Module):
    def __init__(self, in_channels: int, block_params: BlockParams):
        super(RegBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = block_params.out_channels
        self.pool_stride = block_params.pool_stride
        # 输出通道数 = 中间通道数 × expansion
        self.out_channels = self.mid_channels * config.EXPANSION
        self.groups = block_params.groups  # 已在 search_space/encoding 中验证
        self.has_senet = block_params.has_senet == 1
        
        # 新增参数
        self.activation_type = block_params.activation_type
        self.dropout_rate = block_params.dropout_rate
        self.skip_type = block_params.skip_type  # 0=add, 1=concat, 2=none
        self.kernel_size = block_params.kernel_size
        
        # 根据 skip_type 调整输出通道数
        # concat 模式下，最终输出是 identity + out 的拼接
        if self.skip_type == 1:  # concat
            self.final_out_channels = self.out_channels + in_channels
        else:
            self.final_out_channels = self.out_channels
        
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        
        # 使用可变卷积核大小
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
        
        # 根据 skip_type 决定是否需要 shortcut
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
            # concat 模式需要对 identity 进行下采样（如果 stride != 1）
            # 使用与主路径相同的 kernel_size 和 padding 保持一致性
            if block_params.pool_stride != 1:
                self.shortcut = nn.AvgPool2d(kernel_size=3, 
                                              stride=block_params.pool_stride,
                                              padding=1)
            else:
                self.shortcut = nn.Identity()
        else:  # none
            self.shortcut = None
            
        if self.has_senet:
            # SENet 应用于最终输出通道
            se_channels = self.final_out_channels
            # reduction 应该基于实际的 se_channels 计算，确保 se_channels // reduction >= 1
            reduction = min(config.SENET_REDUCTION, se_channels)
            if se_channels // reduction < 1:
                reduction = se_channels  # 确保至少有 1 个隐藏单元
            self.se = SEBlock(se_channels, reduction)
        else:
            self.se = None
        
        # Dropout 层
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_rate)
        else:
            self.dropout = None
            
        # 使用动态激活函数
        self.activation = get_activation(self.activation_type)
        
    def forward(self, x):
        # 处理 identity/shortcut
        if self.skip_type == 0:  # add
            identity = self.shortcut(x)
        elif self.skip_type == 1:  # concat
            identity = self.shortcut(x)
        else:  # none
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
        
        # 根据 skip_type 处理跳跃连接
        if self.skip_type == 0:  # add
            out = out + identity
        elif self.skip_type == 1:  # concat
            out = torch.cat([out, identity], dim=1)
        # skip_type == 2 (none): 不添加跳跃连接
        
        if self.se is not None:
            out = self.se(out)
        
        # 应用 Dropout
        if self.dropout is not None:
            out = self.dropout(out)
            
        out = self.activation(out)
        return out
    
    def get_output_channels(self) -> int:
        return self.final_out_channels

class RegUnit(nn.Module):
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
    
