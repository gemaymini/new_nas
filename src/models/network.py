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
        self.out_channels = in_channels * block_params.pool_stride
        self.groups = min(block_params.groups, self.mid_channels)
        self.has_senet = block_params.has_senet == 1
        
        while self.mid_channels % self.groups != 0:
            self.groups = self.groups // 2 if self.groups > 1 else 1
            
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels,
                               kernel_size=3, stride=1, padding=1, 
                               groups=self.groups, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        
        if block_params.pool_type == 0:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=block_params.pool_stride, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=block_params.pool_stride, padding=1)
            
        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        
        if block_params.pool_stride != 1 or in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels,
                          kernel_size=1, stride=block_params.pool_stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        if self.has_senet:
            reduction = min(config.SENET_REDUCTION, self.out_channels)
            if reduction < 1: reduction = 1
            self.se = SEBlock(self.out_channels, reduction)
        else:
            self.se = None
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + identity
        
        if self.se is not None:
            out = self.se(out)
            
        out = self.relu(out)
        return out
    
    def get_output_channels(self) -> int:
        return self.out_channels

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
    
    @staticmethod
    def test_forward(encoding: List[int], input_size: tuple = None) -> bool:
        if input_size is None:
            input_size = config.NTK_INPUT_SIZE
        try:
            network = NetworkBuilder.build_from_encoding(encoding)
            network.eval()
            x = torch.randn(1, *input_size)
            with torch.no_grad():
                network(x)
            return True
        except Exception as e:
            logger.error(f"Forward test failed: {e}")
            return False
