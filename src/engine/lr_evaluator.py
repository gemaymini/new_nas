# -*- coding: utf-8 -*-
"""
Linear Region Counting (from TE-NAS) for Neural Architecture Search.
Used as a zero-shot proxy for expressivity.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from functools import reduce
from operator import mul

# Add src to path if needed (but it's in src/engine now)
from configuration.config import config

class RandChannel(object):
    """Randomly pick channels from input to increase sample diversity."""
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        if channel < self.num_channel:
            return img
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())


def get_datasets(name, root, input_size, batch_size, num_workers=4):
    """
    Get dataloader for linear region estimation.
    
    Args:
        name: 'cifar10', 'cifar100', or 'imagenet'
        root: Data root directory
        input_size: tuple (C, H, W) or (B, C, H, W)
        batch_size: Batch size
        num_workers: Number of workers
    """
    if len(input_size) == 4:
        input_size = input_size[1:]
    
    # Define transforms based on dataset name
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform = transforms.Compose([
            transforms.RandomCrop(input_size[1], padding=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandChannel(input_size[0])
        ])
        dataset = dset.CIFAR10(root, train=True, transform=transform, download=True)
    
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
        transform = transforms.Compose([
            transforms.RandomCrop(input_size[1], padding=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandChannel(input_size[0])
        ])
        dataset = dset.CIFAR100(root, train=True, transform=transform, download=True)
        
    elif name == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(input_size[1], padding=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandChannel(input_size[0])
        ])
        train_dir = os.path.join(config.IMAGENET_ROOT, 'train')
        if not os.path.exists(train_dir):
             # Fallback to root/val or raise if not found
             if os.path.exists(os.path.join(config.IMAGENET_ROOT, 'val')):
                 train_dir = os.path.join(config.IMAGENET_ROOT, 'val')
             else:
                 raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")
        dataset = dset.ImageFolder(train_dir, transform=transform)
        
    else:
        raise ValueError(f"Unknown dataset: {name}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(config.DEVICE == 'cuda'), drop_last=True)
    return loader


class LinearRegionCount(object):
    """Computes and stores unique activation patterns across multiple samples."""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).to(config.DEVICE)
        
        # Binary activation pattern (ReLU > 0)
        self.activations[self.ptr:self.ptr+n_batch] = (activations > 0).float()
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        if self.activations is None:
             self.n_LR = 0
             return

        # Use half precision if on GPU to save memory and speed up matmul
        if config.DEVICE == 'cuda':
            act = self.activations.half()
        else:
            act = self.activations
            
        # Matrix of Hamming distances (effectively)
        # res[i,j] = sum(act[i] & ~act[j])
        res = torch.matmul(act, (1 - act).T) 
        # Symmetric comparison: res[i,j] = sum(act[i] ^ act[j])
        res += res.T 
        
        # Identical patterns have 0 distance
        identical = (res == 0).float()
        # Count identical patterns for each sample
        counts = identical.sum(1) 
        # Contribution is 1/N where N is the number of identical patterns
        self.n_LR = (1.0 / counts).sum().item()
        
        del self.activations, res, identical, counts
        self.activations = None
        torch.cuda.empty_cache()

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class LREvaluator:
    """Collects linear region counts for a list of models."""
    def __init__(self, models=[], input_size=(3, 32, 32), batch_size=64, sample_batch=10, dataset='cifar10', data_path=None, seed=0):
        self.models = []
        self.input_size = input_size  # CHW
        self.batch_size = batch_size
        self.sample_batch = sample_batch 
        self.dataset = dataset
        self.data_path = data_path or config.DATA_DIR
        self.seed = seed
        self.loader = None
        self.interFeature = []
        self.LRCounts = []
        self.reinit(models, input_size, batch_size, sample_batch, seed)

    def reinit(self, models=None, input_size=None, batch_size=None, sample_batch=None, seed=None):
        if models is not None:
            assert isinstance(models, list)
            self.models = models
            for model in self.models:
                self.register_hook(model)
        
        if input_size is not None:
            self.input_size = input_size
        if batch_size is not None:
            self.batch_size = batch_size
        if sample_batch is not None:
            self.sample_batch = sample_batch
            
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
        if self.loader is None:
             self.loader = get_datasets(self.dataset, self.data_path, self.input_size, self.batch_size)
             self.loader_iter = iter(self.loader)

        total_samples = self.batch_size * self.sample_batch
        self.LRCounts = [LinearRegionCount(total_samples) for _ in range(len(self.models))]

        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        total_samples = self.batch_size * self.sample_batch
        self.LRCounts = [LinearRegionCount(total_samples) for _ in range(len(self.models))]
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach()) 

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = next(self.loader_iter)
            except StopIteration:
                self.loader_iter = iter(self.loader)
                inputs, targets = next(self.loader_iter)
            
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.eval()
            model.to(config.DEVICE)
            model(input_data.to(config.DEVICE))
            
            if len(self.interFeature) == 0: 
                return
            
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)