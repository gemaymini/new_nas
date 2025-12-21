# -*- coding: utf-8 -*-
"""
单元测试 - 评估器模块 (evaluator.py)
测试 NTKEvaluator, FinalEvaluator, FitnessEvaluator 类
"""
import sys
import os
import unittest
import torch
import torch.nn as nn

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from engine.evaluator import NTKEvaluator, clear_gpu_memory, FitnessEvaluator
from models.network import SearchedNetwork, NetworkBuilder
from core.encoding import Individual
from configuration.config import config


class SimpleNet(nn.Module):
    """用于测试的简单网络"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class TestClearGpuMemory(unittest.TestCase):
    """测试 GPU 内存清理函数"""
    
    def test_clear_gpu_memory(self):
        """测试清理 GPU 内存不会报错"""
        # 这个函数应该总是能成功执行，无论是否有 GPU
        try:
            clear_gpu_memory()
        except Exception as e:
            self.fail(f"clear_gpu_memory raised exception: {e}")


class TestNTKEvaluator(unittest.TestCase):
    """测试 NTKEvaluator 类"""
    
    def setUp(self):
        """设置测试"""
        # 使用 CPU 和小批量进行快速测试
        self.evaluator = NTKEvaluator(
            batch_size=4,
            device='cpu',
            num_batch=1
        )
        
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
        self.assertIsInstance(self.evaluator, NTKEvaluator)
        self.assertEqual(self.evaluator.device, 'cpu')
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        evaluator = NTKEvaluator()
        
        self.assertEqual(evaluator.input_size, config.NTK_INPUT_SIZE)
        self.assertEqual(evaluator.num_classes, config.NTK_NUM_CLASSES)
    
    def test_compute_ntk_score_simple_network(self):
        """测试简单网络的 NTK 分数计算"""
        network = SimpleNet()
        param_count = network.get_param_count()
        
        score = self.evaluator.compute_ntk_score(network, param_count)
        
        self.assertIsInstance(score, float)
        # NTK 条件数应该是正数
        self.assertGreater(score, 0)
    
    def test_compute_ntk_score_exceeds_threshold(self):
        """测试超过参数阈值的网络"""
        network = SimpleNet()
        # 使用一个非常小的阈值
        evaluator = NTKEvaluator(batch_size=4, device='cpu')
        evaluator.param_threshold = 1  # 非常小的阈值
        
        score = evaluator.compute_ntk_score(network, param_count=1000)
        
        # 应该返回惩罚值
        self.assertEqual(score, 100000.0)
    
    def test_evaluate_individual(self):
        """测试评估个体"""
        ind = Individual(self.valid_encoding)
        ind.id = 1
        
        fitness = self.evaluator.evaluate_individual(ind)
        
        self.assertIsInstance(fitness, float)
        # fitness 是负的条件数
        self.assertLessEqual(fitness, 0)
        
        # 检查个体属性已被设置
        self.assertIsNotNone(ind.fitness)
        self.assertIsNotNone(ind.param_count)
    
    def test_compute_ntk_condition_number(self):
        """测试计算 NTK 条件数"""
        network = SimpleNet()
        network.to(self.evaluator.device)
        
        cond = self.evaluator.compute_ntk_condition_number(
            network, self.evaluator.trainloader, num_batch=1
        )
        
        self.assertIsInstance(cond, float)
        self.assertGreater(cond, 0)


class TestFitnessEvaluator(unittest.TestCase):
    """测试 FitnessEvaluator 类"""
    
    def setUp(self):
        """设置测试"""
        self.evaluator = FitnessEvaluator()
        
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
        self.assertIsInstance(self.evaluator, FitnessEvaluator)
        self.assertIsNotNone(self.evaluator.ntk_evaluator)
    
    def test_evaluate_individual(self):
        """测试评估个体"""
        ind = Individual(self.valid_encoding)
        ind.id = 1
        
        fitness = self.evaluator.evaluate_individual(ind)
        
        self.assertIsInstance(fitness, float)
        self.assertEqual(fitness, ind.fitness)


class TestNTKRecalBN(unittest.TestCase):
    """测试 NTK BN 重新计算功能"""
    
    def setUp(self):
        """设置测试"""
        self.evaluator = NTKEvaluator(
            batch_size=4,
            device='cpu',
            recalbn=2,
            num_batch=1
        )
    
    def test_recal_bn(self):
        """测试 BN 重新计算"""
        network = SimpleNet()
        
        # 调用 recal_bn
        network = self.evaluator.recal_bn(
            network, self.evaluator.trainloader, 
            recal_batches=2, device='cpu'
        )
        
        # 网络应该仍然可以正常工作
        x = torch.randn(2, 3, 32, 32)
        output = network(x)
        
        self.assertEqual(output.shape[0], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
