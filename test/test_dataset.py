# -*- coding: utf-8 -*-
"""
单元测试 - 数据集模块 (dataset.py)
测试 DatasetLoader 类
"""
import sys
import os
import unittest
import torch

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data.dataset import DatasetLoader
from configuration.config import config


class TestDatasetLoaderCIFAR10(unittest.TestCase):
    """测试 CIFAR-10 数据集加载"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 加载数据集一次"""
        # 使用小 batch_size 加速测试
        cls.trainloader, cls.testloader = DatasetLoader.get_cifar10(batch_size=16, num_workers=0)
    
    def test_trainloader_not_none(self):
        """测试训练数据加载器不为空"""
        self.assertIsNotNone(self.trainloader)
    
    def test_testloader_not_none(self):
        """测试测试数据加载器不为空"""
        self.assertIsNotNone(self.testloader)
    
    def test_trainloader_batch(self):
        """测试训练数据批次"""
        for inputs, targets in self.trainloader:
            self.assertEqual(inputs.shape[0], 16)  # batch_size
            self.assertEqual(inputs.shape[1], 3)   # channels
            self.assertEqual(inputs.shape[2], 32)  # height
            self.assertEqual(inputs.shape[3], 32)  # width
            self.assertEqual(targets.shape[0], 16)
            break
    
    def test_testloader_batch(self):
        """测试测试数据批次"""
        for inputs, targets in self.testloader:
            self.assertEqual(inputs.shape[1], 3)
            self.assertEqual(inputs.shape[2], 32)
            self.assertEqual(inputs.shape[3], 32)
            break
    
    def test_trainloader_labels_range(self):
        """测试训练数据标签范围"""
        for _, targets in self.trainloader:
            self.assertTrue(torch.all(targets >= 0))
            self.assertTrue(torch.all(targets < 10))
            break
    
    def test_trainloader_normalized(self):
        """测试训练数据已归一化"""
        for inputs, _ in self.trainloader:
            # 归一化后数据应该在合理范围内
            self.assertTrue(inputs.min() >= -3.0)
            self.assertTrue(inputs.max() <= 3.0)
            break


class TestDatasetLoaderNTK(unittest.TestCase):
    """测试 NTK 数据加载器"""
    
    def test_get_ntk_trainloader_cifar10(self):
        """测试 CIFAR-10 NTK 数据加载器"""
        loader = DatasetLoader.get_ntk_trainloader(batch_size=8, num_workers=0, dataset_name='cifar10')
        
        self.assertIsNotNone(loader)
        
        for inputs, targets in loader:
            self.assertEqual(inputs.shape[0], 8)
            self.assertEqual(inputs.shape[1], 3)
            self.assertEqual(inputs.shape[2], 32)
            self.assertEqual(inputs.shape[3], 32)
            break
    
    def test_get_ntk_trainloader_cifar100(self):
        """测试 CIFAR-100 NTK 数据加载器"""
        loader = DatasetLoader.get_ntk_trainloader(batch_size=8, num_workers=0, dataset_name='cifar100')
        
        self.assertIsNotNone(loader)
        
        for inputs, targets in loader:
            self.assertEqual(inputs.shape[0], 8)
            self.assertTrue(torch.all(targets >= 0))
            self.assertTrue(torch.all(targets < 100))
            break
    
    def test_get_ntk_trainloader_invalid_dataset(self):
        """测试无效数据集名称"""
        with self.assertRaises(ValueError):
            DatasetLoader.get_ntk_trainloader(dataset_name='invalid')


class TestDatasetLoaderCIFAR100(unittest.TestCase):
    """测试 CIFAR-100 数据集加载"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 加载数据集一次"""
        cls.trainloader, cls.testloader = DatasetLoader.get_cifar100(batch_size=16, num_workers=0)
    
    def test_trainloader_not_none(self):
        """测试训练数据加载器不为空"""
        self.assertIsNotNone(self.trainloader)
    
    def test_testloader_not_none(self):
        """测试测试数据加载器不为空"""
        self.assertIsNotNone(self.testloader)
    
    def test_trainloader_labels_range(self):
        """测试训练数据标签范围"""
        for _, targets in self.trainloader:
            self.assertTrue(torch.all(targets >= 0))
            self.assertTrue(torch.all(targets < 100))
            break


class TestDatasetLoaderDefaultParams(unittest.TestCase):
    """测试默认参数"""
    
    def test_cifar10_default_batch_size(self):
        """测试 CIFAR-10 默认批次大小"""
        trainloader, _ = DatasetLoader.get_cifar10(num_workers=0)
        
        for inputs, _ in trainloader:
            self.assertEqual(inputs.shape[0], config.BATCH_SIZE)
            break
    
    def test_cifar100_default_batch_size(self):
        """测试 CIFAR-100 默认批次大小"""
        trainloader, _ = DatasetLoader.get_cifar100(num_workers=0)
        
        for inputs, _ in trainloader:
            self.assertEqual(inputs.shape[0], config.BATCH_SIZE)
            break


if __name__ == '__main__':
    unittest.main(verbosity=2)
