# -*- coding: utf-8 -*-
"""
Dataset loaders and preprocessing pipelines.
"""
import torch
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from configuration.config import config


class DatasetLoader:
    """Factory for common dataset loaders."""

    @staticmethod
    def get_cifar10(batch_size: int = None, num_workers: int = None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if num_workers is None:
            num_workers = config.NUM_WORKERS

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

        root = config.DATA_DIR
        download = not os.path.exists(os.path.join(root, "cifar-10-batches-py"))

        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return trainloader, testloader

    @staticmethod
    def get_imagenet(batch_size: int = None, num_workers: int = None):
        """Return ImageNet train/val DataLoaders."""
        if batch_size is None:
            batch_size = config.IMAGENET_BATCH_SIZE
        if num_workers is None:
            num_workers = config.NUM_WORKERS

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGENET_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            normalize,
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGENET_INPUT_SIZE),
            transforms.ToTensor(),
            normalize,
        ])

        root = config.IMAGENET_ROOT
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"ImageNet train dir not found: {train_dir}\n"
                f"Expected layout:\n"
                f"  {root}/train/n01440764/... \n"
                f"  {root}/val/n01440764/..."
            )

        trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        valset = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)
        valloader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return trainloader, valloader

    @staticmethod
    def get_ntk_trainloader(
        batch_size: int = 64,
        num_workers: int = 0,
        dataset_name: str = "cifar10",
    ):
        """Return a small-data loader for NTK estimation."""
        root = config.DATA_DIR

        if dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            download = not os.path.exists(os.path.join(root, "cifar-10-batches-py"))
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=download, transform=transform
            )
        elif dataset_name == "cifar100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
            ])
            download = not os.path.exists(os.path.join(root, "cifar-100-python"))
            dataset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=download, transform=transform
            )
        elif dataset_name == "imagenet":
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(config.IMAGENET_INPUT_SIZE),
                transforms.ToTensor(),
                normalize,
            ])
            train_dir = os.path.join(config.IMAGENET_ROOT, "train")
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")
            dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        return loader

    @staticmethod
    def get_cifar100(batch_size: int = None, num_workers: int = None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if num_workers is None:
            num_workers = config.NUM_WORKERS

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

        root = config.DATA_DIR
        download = not os.path.exists(os.path.join(root, "cifar-100-python"))

        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=download, transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=download, transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return trainloader, testloader
