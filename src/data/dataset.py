# -*- coding: utf-8 -*-
"""
数据集加载模块
负责数据集的下载、预处理和加载
"""
import torch
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from configuration.config import config

class DatasetLoader:
    """
    数据集加载器
    """
    @staticmethod
    def get_cifar10(batch_size: int = None, num_workers: int = None):
        if batch_size is None: batch_size = config.BATCH_SIZE
        if num_workers is None: num_workers = config.NUM_WORKERS
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
        
        root = './data'
        download = True
        if os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
            download = False
            
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform_test)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return trainloader, testloader

    @staticmethod
    def get_ntk_trainloader(batch_size: int = 16, num_workers: int = 0, dataset_name: str = 'cifar10'):
        """
        获取用于NTK计算的DataLoader
        通常使用较小的batch_size，且不进行数据增强
        """
        root = './data'
        
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010)),
            ])
            download = True
            if os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
                download = False
                
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=download, transform=transform)
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                   (0.2675, 0.2565, 0.2761)),
            ])
            download = True
            if os.path.exists(os.path.join(root, 'cifar-100-python')):
                download = False
                
            dataset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=download, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        return loader

    @staticmethod
    def get_cifar100(batch_size: int = None, num_workers: int = None):
        if batch_size is None: batch_size = config.BATCH_SIZE
        if num_workers is None: num_workers = config.NUM_WORKERS
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761)),
        ])
        
        root = './data'
        download = True
        if os.path.exists(os.path.join(root, 'cifar-100-python')):
            download = False
            
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=download, transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=download, transform=transform_test)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return trainloader, testloader
