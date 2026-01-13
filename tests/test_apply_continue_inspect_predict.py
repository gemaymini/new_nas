# -*- coding: utf-8 -*-
"""
Tests for apply helpers: continue_train, inspect_model, predict.
"""
import io
from pathlib import Path

import torch
from PIL import Image

from apply.continue_train import continue_training
from apply.inspect_model import inspect_model
from apply.predict import predict_image
from configuration.config import config
from core.encoding import Individual
from models.network import NetworkBuilder

from conftest import make_encoding


def _make_checkpoint(path: Path, encoding):
    net = NetworkBuilder.build_from_encoding(encoding, input_channels=3, num_classes=10)
    state = net.state_dict()
    checkpoint = {
        "state_dict": state,
        "encoding": encoding,
        "accuracy": 12.34,
        "param_count": net.get_param_count(),
        "history": [],
    }
    torch.save(checkpoint, path)


def _make_image(path: Path):
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(path)


def test_continue_training(monkeypatch, tmp_path):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    model_path = tmp_path / "model.pth"
    _make_checkpoint(model_path, encoding)

    def fake_train(*args, **kwargs):
        history = [{"epoch": 1, "train_acc": 10.0, "test_acc": 12.34}]
        return 12.34, history

    monkeypatch.setattr(config, "FINAL_DATASET", "cifar10", raising=False)
    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr("apply.continue_train.DatasetLoader.get_cifar10", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr("apply.continue_train.NetworkTrainer.train_network", fake_train)

    continue_training(str(model_path), epochs=1, lr=0.01)
    saved = list(tmp_path.glob("*_continued_acc*.pth"))
    assert saved


def test_inspect_model(capsys, tmp_path):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    model_path = tmp_path / "model.pth"
    _make_checkpoint(model_path, encoding)
    inspect_model(str(model_path))
    out = capsys.readouterr().out
    assert "INFO: encoding=" in out


def test_predict_image(capsys, tmp_path, monkeypatch):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    model_path = tmp_path / "model.pth"
    _make_checkpoint(model_path, encoding)
    image_path = tmp_path / "image.png"
    _make_image(image_path)
    monkeypatch.setattr("apply.predict.transforms.Compose", lambda *args, **kwargs: (lambda img: torch.zeros(3, 32, 32)))
    predict_image(str(model_path), str(image_path), encoding_str=None, device="cpu")
    out = capsys.readouterr().out
    assert "INFO: prediction" in out
