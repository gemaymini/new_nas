# -*- coding: utf-8 -*-
"""
Tests for apply.retrain_model.
"""
from pathlib import Path

import torch

from apply.retrain_model import retrain_model
from configuration.config import config
from models.network import NetworkBuilder

from conftest import make_encoding


def _make_checkpoint(path: Path, encoding):
    net = NetworkBuilder.build_from_encoding(encoding, input_channels=3, num_classes=10)
    checkpoint = {
        "state_dict": net.state_dict(),
        "encoding": encoding,
        "history": [],
    }
    torch.save(checkpoint, path)


def test_retrain_model(monkeypatch, tmp_path):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    model_path = tmp_path / "model.pth"
    _make_checkpoint(model_path, encoding)

    def fake_train(*args, **kwargs):
        history = [{"epoch": 1, "train_acc": 10.0, "test_acc": 11.0}]
        return 11.0, history

    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr("apply.retrain_model.DatasetLoader.get_cifar10", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr("apply.retrain_model.NetworkTrainer.train_network", fake_train)

    summary = retrain_model(
        model_path=str(model_path),
        epochs=1,
        num_runs=1,
        dataset="cifar10",
        device="cpu",
        save_results=False,
    )
    assert summary["status"] == "completed"
