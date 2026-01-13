# -*- coding: utf-8 -*-
"""
Tests for engine.trainer.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from engine.trainer import NetworkTrainer


def _make_loader(num_samples=8, num_classes=2):
    inputs = torch.randn(num_samples, 1, 2, 2)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_train_one_epoch():
    trainer = NetworkTrainer(device="cpu")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    loader = _make_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, acc = trainer.train_one_epoch(model, loader, criterion, optimizer, epoch=1, total_epochs=1)
    assert loss > 0
    assert 0 <= acc <= 100


def test_evaluate():
    trainer = NetworkTrainer(device="cpu")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    loader = _make_loader()
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = trainer.evaluate(model, loader, criterion)
    assert loss > 0
    assert 0 <= acc <= 100


def test_train_network():
    trainer = NetworkTrainer(device="cpu")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    trainloader = _make_loader()
    testloader = _make_loader()
    best_acc, history = trainer.train_network(
        model,
        trainloader,
        testloader,
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        early_stopping=False,
    )
    assert len(history) == 1
    assert best_acc >= 0
