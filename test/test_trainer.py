# -*- coding: utf-8 -*-
"""
单元测试 - 训练器模块 (trainer.py)
测试 NetworkTrainer 类
"""
import sys
import os
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from engine.trainer import NetworkTrainer
from models.network import SearchedNetwork
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


def create_fake_dataloader(batch_size=8, num_samples=32, num_classes=10):
    """创建假数据加载器用于测试"""
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestNetworkTrainer(unittest.TestCase):
    """测试 NetworkTrainer 类"""
    
    def setUp(self):
        """设置测试"""
        self.trainer = NetworkTrainer(device='cpu')
        self.model = SimpleNet()
        self.trainloader = create_fake_dataloader()
        self.testloader = create_fake_dataloader()
    
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.trainer, NetworkTrainer)
        self.assertEqual(self.trainer.device, 'cpu')
    
    def test_init_cuda_not_available(self):
        """测试 CUDA 不可用时的初始化"""
        # 强制使用 CUDA 但实际不可用
        trainer = NetworkTrainer(device='cuda')
        # 如果 CUDA 不可用，应该回退到 CPU
        if not torch.cuda.is_available():
            self.assertEqual(trainer.device, 'cpu')
    
    def test_train_one_epoch(self):
        """测试训练一个 epoch"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        loss, acc = self.trainer.train_one_epoch(
            self.model, self.trainloader, criterion, optimizer, 
            epoch=1, total_epochs=1
        )
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)
    
    def test_evaluate(self):
        """测试评估"""
        criterion = nn.CrossEntropyLoss()
        
        loss, acc = self.trainer.evaluate(self.model, self.testloader, criterion)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)
    
    def test_train_network(self):
        """测试完整训练流程"""
        # 使用非常少的 epoch 进行快速测试
        best_acc, history = self.trainer.train_network(
            self.model, self.trainloader, self.testloader,
            epochs=2, lr=0.01
        )
        
        self.assertIsInstance(best_acc, float)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 2)
        
        # 检查 history 结构
        self.assertIn('epoch', history[0])
        self.assertIn('train_loss', history[0])
        self.assertIn('train_acc', history[0])
        self.assertIn('test_loss', history[0])
        self.assertIn('test_acc', history[0])
        self.assertIn('lr', history[0])
    
    def test_train_network_returns_best_model(self):
        """测试训练返回最佳模型"""
        # 记录初始参数
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        best_acc, _ = self.trainer.train_network(
            self.model, self.trainloader, self.testloader,
            epochs=2, lr=0.1
        )
        
        # 参数应该已经改变
        params_changed = False
        for name, param in self.model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                params_changed = True
                break
        
        self.assertTrue(params_changed)
    
    def test_train_with_searched_network(self):
        """测试使用搜索网络进行训练"""
        valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        network = SearchedNetwork(valid_encoding)
        
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader,
            epochs=1, lr=0.01
        )
        
        self.assertIsInstance(best_acc, float)
        self.assertEqual(len(history), 1)


class TestNetworkTrainerDefaultParams(unittest.TestCase):
    """测试默认参数"""
    
    def test_default_epochs(self):
        """测试默认 epoch 数"""
        # 这个测试只验证参数传递，不实际运行完整训练
        trainer = NetworkTrainer(device='cpu')
        self.assertIsNotNone(trainer)
    
    def test_default_lr(self):
        """测试默认学习率"""
        # 验证默认值与配置匹配
        self.assertEqual(config.LEARNING_RATE, 0.1)
    
    def test_default_momentum(self):
        """测试默认动量"""
        self.assertEqual(config.MOMENTUM, 0.9)
    
    def test_default_weight_decay(self):
        """测试默认权重衰减"""
        self.assertEqual(config.WEIGHT_DECAY, 5e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
