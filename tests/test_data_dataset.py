# -*- coding: utf-8 -*-
"""
Tests for data.dataset.
"""
import os

import torch

from configuration.config import config
from data.dataset import DatasetLoader


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.zeros(3, 32, 32), 0


def test_get_cifar10(monkeypatch):
    monkeypatch.setattr("torchvision.datasets.CIFAR10", DummyDataset)
    trainloader, testloader = DatasetLoader.get_cifar10(batch_size=2, num_workers=0)
    assert len(trainloader) > 0
    assert len(testloader) > 0


def test_get_cifar100(monkeypatch):
    monkeypatch.setattr("torchvision.datasets.CIFAR100", DummyDataset)
    trainloader, testloader = DatasetLoader.get_cifar100(batch_size=2, num_workers=0)
    assert len(trainloader) > 0
    assert len(testloader) > 0


def test_get_imagenet(monkeypatch, tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "IMAGENET_ROOT", str(tmp_path), raising=False)
    monkeypatch.setattr("torchvision.datasets.ImageFolder", DummyDataset)
    trainloader, valloader = DatasetLoader.get_imagenet(batch_size=2, num_workers=0)
    assert len(trainloader) > 0
    assert len(valloader) > 0


def test_get_ntk_trainloader_cifar10(monkeypatch):
    monkeypatch.setattr("torchvision.datasets.CIFAR10", DummyDataset)
    loader = DatasetLoader.get_ntk_trainloader(batch_size=2, dataset_name="cifar10")
    assert len(loader) > 0


def test_get_ntk_trainloader_cifar100(monkeypatch):
    monkeypatch.setattr("torchvision.datasets.CIFAR100", DummyDataset)
    loader = DatasetLoader.get_ntk_trainloader(batch_size=2, dataset_name="cifar100")
    assert len(loader) > 0


def test_get_ntk_trainloader_imagenet(monkeypatch, tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "IMAGENET_ROOT", str(tmp_path), raising=False)
    monkeypatch.setattr("torchvision.datasets.ImageFolder", DummyDataset)
    loader = DatasetLoader.get_ntk_trainloader(batch_size=2, dataset_name="imagenet")
    assert len(loader) > 0


def test_get_ntk_trainloader_invalid():
    try:
        DatasetLoader.get_ntk_trainloader(batch_size=2, dataset_name="unknown")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
