# -*- coding: utf-8 -*-
"""
Tests for engine.evaluator.
"""
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.config import config
from core.encoding import Individual
from engine.evaluator import NTKEvaluator, FinalEvaluator, FitnessEvaluator
from models.network import NetworkBuilder

from conftest import make_encoding


def _dummy_loader(num_samples=4, num_classes=2):
    inputs = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_ntk_condition_number(monkeypatch):
    monkeypatch.setattr(
        "engine.evaluator.DatasetLoader.get_ntk_trainloader",
        lambda *args, **kwargs: _dummy_loader(),
    )
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    net = NetworkBuilder.build_from_encoding(encoding, input_channels=3, num_classes=2)
    evaluator = NTKEvaluator(device="cpu", num_batch=1, batch_size=2, dataset="cifar10")
    cond = evaluator.compute_ntk_condition_number(net, evaluator.trainloader, num_batch=1)
    assert cond > 0


def test_ntk_score_with_patch(monkeypatch):
    monkeypatch.setattr(
        "engine.evaluator.DatasetLoader.get_ntk_trainloader",
        lambda *args, **kwargs: _dummy_loader(),
    )
    evaluator = NTKEvaluator(device="cpu", num_batch=1, batch_size=2, dataset="cifar10")
    monkeypatch.setattr(evaluator, "compute_ntk_condition_number", lambda *args, **kwargs: 10.0)
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    net = NetworkBuilder.build_from_encoding(encoding, input_channels=3, num_classes=2)
    score = evaluator.compute_ntk_score(net, param_count=10, num_runs=3)
    assert score == 10.0


def test_evaluate_individual_sets_fitness(monkeypatch):
    monkeypatch.setattr(
        "engine.evaluator.DatasetLoader.get_ntk_trainloader",
        lambda *args, **kwargs: _dummy_loader(),
    )
    evaluator = NTKEvaluator(device="cpu", num_batch=1, batch_size=2, dataset="cifar10")
    monkeypatch.setattr(evaluator, "compute_ntk_score", lambda *args, **kwargs: 7.0)
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    ind.id = 1
    score = evaluator.evaluate_individual(ind)
    assert score == 7.0
    assert ind.fitness == 7.0


def test_plot_training_history(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.evaluator.DatasetLoader.get_cifar10", lambda *args, **kwargs: (_dummy_loader(), _dummy_loader()))
    evaluator = FinalEvaluator(dataset="cifar10", device="cpu")
    history = [
        {"epoch": 1, "train_loss": 1.0, "test_loss": 1.2, "train_acc": 50.0, "test_acc": 45.0, "lr": 0.01},
        {"epoch": 2, "train_loss": 0.9, "test_loss": 1.1, "train_acc": 55.0, "test_acc": 50.0, "lr": 0.01},
    ]
    out = evaluator.plot_training_history(
        history=history,
        individual_id=1,
        epochs=2,
        best_acc=50.0,
        param_count=1234,
        save_dir=str(tmp_path),
    )
    assert out is not None


def test_evaluate_individual(monkeypatch, tmp_path):
    monkeypatch.setattr("engine.evaluator.DatasetLoader.get_cifar10", lambda *args, **kwargs: (_dummy_loader(), _dummy_loader()))
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path), raising=False)

    def fake_train(*args, **kwargs):
        history = [{"epoch": 1, "train_loss": 1.0, "test_loss": 1.0, "train_acc": 10.0, "test_acc": 20.0, "lr": 0.01}]
        return 20.0, history

    monkeypatch.setattr("engine.evaluator.NetworkTrainer.train_network", fake_train)
    monkeypatch.setattr("engine.evaluator.FinalEvaluator.plot_training_history", lambda *args, **kwargs: None)

    evaluator = FinalEvaluator(dataset="cifar10", device="cpu")
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    ind.id = 1
    best_acc, result = evaluator.evaluate_individual(ind, epochs=1)
    assert best_acc == 20.0
    assert result["individual_id"] == 1


def test_evaluate_top_individuals(monkeypatch):
    monkeypatch.setattr("engine.evaluator.DatasetLoader.get_cifar10", lambda *args, **kwargs: (_dummy_loader(), _dummy_loader()))
    evaluator = FinalEvaluator(dataset="cifar10", device="cpu")

    def fake_eval(ind, epochs=None):
        return ind.quick_score, {"individual_id": ind.id}

    monkeypatch.setattr(evaluator, "evaluate_individual", fake_eval)
    pop = []
    for i in range(3):
        ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
        ind.id = i
        ind.fitness = i
        ind.quick_score = 10.0 + i
        pop.append(ind)
    best_ind, results = evaluator.evaluate_top_individuals(population=pop, top_k=2, epochs=1)
    assert best_ind is not None
    assert len(results) == 2


def test_fitness_evaluator_population(monkeypatch):
    fe = FitnessEvaluator()
    monkeypatch.setattr("engine.evaluator.NTKEvaluator", lambda *args, **kwargs: fe._ntk_evaluator)
    fe._ntk_evaluator = type(
        "DummyEvaluator",
        (),
        {"evaluate_individual": lambda self, ind: setattr(ind, "fitness", 1.0)},
    )()

    pop = []
    for i in range(2):
        ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
        ind.id = i
        pop.append(ind)
    fe.evaluate_population_ntk(pop, show_progress=False)
    assert all(ind.fitness == 1.0 for ind in pop)
