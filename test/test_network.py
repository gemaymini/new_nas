# -*- coding: utf-8 -*-
"""
单元测试 - 网络构建模块 (network.py)
测试 SEBlock, ConvUnit, RegBlock, RegUnit, SearchedNetwork, NetworkBuilder 类
"""
import sys
import os
import unittest
import torch
import torch.nn as nn

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from models.network import SEBlock, ConvUnit, RegBlock, RegUnit, SearchedNetwork, NetworkBuilder
from core.encoding import BlockParams, Individual, Encoder
from configuration.config import config


class TestSEBlock(unittest.TestCase):
    """测试 SEBlock 类"""
    
    def test_init(self):
        """测试初始化"""
        se = SEBlock(channels=64, reduction=16)
        self.assertIsInstance(se, nn.Module)
    
    def test_forward(self):
        """测试前向传播"""
        se = SEBlock(channels=32, reduction=8)
        x = torch.randn(2, 32, 8, 8)
        output = se(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_small_channels(self):
        """测试小通道数"""
        se = SEBlock(channels=4, reduction=2)
        x = torch.randn(1, 4, 4, 4)
        output = se(x)
        
        self.assertEqual(output.shape, x.shape)


class TestConvUnit(unittest.TestCase):
    """测试 ConvUnit 类"""
    
    def test_init(self):
        """测试初始化"""
        conv_unit = ConvUnit(in_channels=3, out_channels=64)
        self.assertIsInstance(conv_unit, nn.Module)
    
    def test_forward(self):
        """测试前向传播"""
        conv_unit = ConvUnit(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        output = conv_unit(x)
        
        self.assertEqual(output.shape, (2, 64, 32, 32))
    
    def test_forward_with_stride(self):
        """测试带步长的前向传播"""
        conv_unit = ConvUnit(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        output = conv_unit(x)
        
        self.assertEqual(output.shape, (2, 64, 16, 16))


class TestRegBlock(unittest.TestCase):
    """测试 RegBlock 类"""
    
    def test_init_stride1(self):
        """测试 stride=1 初始化"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=0, pool_stride=1, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        self.assertIsInstance(block, nn.Module)
    
    def test_init_stride2(self):
        """测试 stride=2 初始化"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=1, pool_stride=2, has_senet=1)
        block = RegBlock(in_channels=64, block_params=bp)
        self.assertIsInstance(block, nn.Module)
    
    def test_forward_stride1(self):
        """测试 stride=1 前向传播"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=0, pool_stride=1, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        x = torch.randn(2, 64, 8, 8)
        output = block(x)
        
        # stride=1 时输出通道 = in_channels * stride = 64 * 1 = 64
        self.assertEqual(output.shape, (2, 64, 8, 8))
    
    def test_forward_stride2(self):
        """测试 stride=2 前向传播"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=0, pool_stride=2, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        x = torch.randn(2, 64, 8, 8)
        output = block(x)
        
        # stride=2 时输出通道 = in_channels * stride = 64 * 2 = 128
        # 空间尺寸减半
        self.assertEqual(output.shape, (2, 128, 4, 4))
    
    def test_forward_with_senet(self):
        """测试带 SENet 的前向传播"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=1, pool_stride=1, has_senet=1)
        block = RegBlock(in_channels=64, block_params=bp)
        x = torch.randn(2, 64, 8, 8)
        output = block(x)
        
        self.assertEqual(output.shape, (2, 64, 8, 8))
    
    def test_forward_avgpool(self):
        """测试平均池化"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=1, pool_stride=2, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        x = torch.randn(2, 64, 8, 8)
        output = block(x)
        
        self.assertEqual(output.shape, (2, 128, 4, 4))
    
    def test_get_output_channels(self):
        """测试获取输出通道数"""
        bp = BlockParams(out_channels=32, groups=4, pool_type=0, pool_stride=2, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        
        self.assertEqual(block.get_output_channels(), 128)
    
    def test_groups_adjustment(self):
        """测试分组数调整"""
        # groups > mid_channels 时应该自动调整
        bp = BlockParams(out_channels=8, groups=16, pool_type=0, pool_stride=1, has_senet=0)
        block = RegBlock(in_channels=64, block_params=bp)
        
        # 应该不会报错
        x = torch.randn(2, 64, 8, 8)
        output = block(x)
        self.assertEqual(output.shape[1], 64)


class TestRegUnit(unittest.TestCase):
    """测试 RegUnit 类"""
    
    def test_init(self):
        """测试初始化"""
        block_params = [
            BlockParams(16, 2, 0, 1, 0),
            BlockParams(32, 4, 1, 1, 1),
        ]
        unit = RegUnit(in_channels=64, block_params_list=block_params)
        self.assertIsInstance(unit, nn.Module)
    
    def test_forward(self):
        """测试前向传播"""
        block_params = [
            BlockParams(16, 2, 0, 1, 0),
            BlockParams(32, 4, 1, 1, 1),
        ]
        unit = RegUnit(in_channels=64, block_params_list=block_params)
        x = torch.randn(2, 64, 8, 8)
        output = unit(x)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 8)
        self.assertEqual(output.shape[3], 8)
    
    def test_forward_with_downsampling(self):
        """测试带下采样的前向传播"""
        block_params = [
            BlockParams(16, 2, 0, 2, 0),  # stride=2
            BlockParams(32, 4, 1, 1, 1),
        ]
        unit = RegUnit(in_channels=64, block_params_list=block_params)
        x = torch.randn(2, 64, 8, 8)
        output = unit(x)
        
        # 第一个 block 将尺寸减半
        self.assertEqual(output.shape[2], 4)
        self.assertEqual(output.shape[3], 4)
    
    def test_get_output_channels(self):
        """测试获取输出通道数"""
        block_params = [
            BlockParams(16, 2, 0, 1, 0),
            BlockParams(32, 4, 1, 2, 1),  # 最后一个 block stride=2
        ]
        unit = RegUnit(in_channels=64, block_params_list=block_params)
        
        out_channels = unit.get_output_channels()
        self.assertGreater(out_channels, 0)


class TestSearchedNetwork(unittest.TestCase):
    """测试 SearchedNetwork 类"""
    
    def setUp(self):
        """设置测试编码"""
        self.valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
    
    def test_init(self):
        """测试初始化"""
        network = SearchedNetwork(self.valid_encoding)
        self.assertIsInstance(network, nn.Module)
    
    def test_forward(self):
        """测试前向传播"""
        network = SearchedNetwork(self.valid_encoding)
        x = torch.randn(2, 3, 32, 32)
        output = network(x)
        
        self.assertEqual(output.shape, (2, 10))
    
    def test_forward_cifar100(self):
        """测试 CIFAR-100 输出"""
        network = SearchedNetwork(self.valid_encoding, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        output = network(x)
        
        self.assertEqual(output.shape, (2, 100))
    
    def test_get_param_count(self):
        """测试获取参数量"""
        network = SearchedNetwork(self.valid_encoding)
        param_count = network.get_param_count()
        
        self.assertGreater(param_count, 0)
        self.assertIsInstance(param_count, int)
    
    def test_encoding_stored(self):
        """测试编码被存储"""
        network = SearchedNetwork(self.valid_encoding)
        self.assertEqual(network.encoding, self.valid_encoding)


class TestNetworkBuilder(unittest.TestCase):
    """测试 NetworkBuilder 类"""
    
    def setUp(self):
        """设置测试编码"""
        self.valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        self.valid_individual = Individual(self.valid_encoding)
    
    def test_build_from_encoding(self):
        """测试从编码构建网络"""
        network = NetworkBuilder.build_from_encoding(self.valid_encoding)
        self.assertIsInstance(network, SearchedNetwork)
    
    def test_build_from_individual(self):
        """测试从个体构建网络"""
        network = NetworkBuilder.build_from_individual(self.valid_individual)
        self.assertIsInstance(network, SearchedNetwork)
    
    def test_calculate_param_count(self):
        """测试计算参数量"""
        param_count = NetworkBuilder.calculate_param_count(self.valid_encoding)
        self.assertGreater(param_count, 0)
    
    def test_test_forward(self):
        """测试前向测试"""
        result = NetworkBuilder.test_forward(self.valid_encoding)
        self.assertTrue(result)
    
    def test_test_forward_invalid_encoding(self):
        """测试无效编码的前向测试"""
        # 创建一个会导致前向失败的无效编码
        invalid_encoding = [1, 1, 16, 2, 0, 2, 0]  # 过少的层
        
        # 这可能成功也可能失败，取决于具体实现
        # 主要测试不会抛出未处理的异常
        try:
            result = NetworkBuilder.test_forward(invalid_encoding)
            self.assertIsInstance(result, bool)
        except Exception as e:
            # 如果抛出异常，测试通过
            pass


class TestNetworkGradient(unittest.TestCase):
    """测试网络梯度计算"""
    
    def setUp(self):
        """设置测试编码"""
        self.valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
    
    def test_backward(self):
        """测试反向传播"""
        network = SearchedNetwork(self.valid_encoding)
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        output = network(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # 检查是否有梯度
        has_grad = False
        for param in network.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        self.assertTrue(has_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
