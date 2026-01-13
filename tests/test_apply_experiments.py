# -*- coding: utf-8 -*-
"""
Tests for experiment scripts in apply.
"""
import json
from pathlib import Path

import torch

from apply.correlation_experiment import ExperimentLogger as CorrLogger, Visualizer as CorrVisualizer
from apply.ntk_correlation_experiment import ExperimentLogger as NTKLogger, Visualizer as NTKVisualizer
from apply.correlation_experiment import run_correlation_experiment
from apply.ntk_correlation_experiment import run_ntk_experiment
from configuration.config import config
from core.encoding import Individual
from models.network import NetworkBuilder

from conftest import make_encoding


def _dummy_loader():
    data = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 2, (4,))
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)


def test_correlation_logger_and_visualizer(tmp_path):
    logger = CorrLogger(str(tmp_path))
    logger.set_config({"short_epochs": 1, "full_epochs": 2})
    logger.log_model_result(0, [1, 2, 3], [{"epoch": 1, "test_acc": 10.0}], 10.0, 20.0)
    logger.log_model_result(1, [4, 5, 6], [{"epoch": 1, "test_acc": 12.0}], 12.0, 22.0)
    log_file = logger.finish()
    assert Path(log_file).exists()

    viz = CorrVisualizer(log_file, str(tmp_path))
    viz.generate_all()
    assert (tmp_path / "correlation_plot.png").exists()
    assert (tmp_path / "training_curves.png").exists()


def test_ntk_logger_and_visualizer(tmp_path):
    logger = NTKLogger(str(tmp_path))
    logger.set_config({"short_epochs": 1})
    logger.log_model_result(0, [1, 2, 3], [{"epoch": 1, "test_acc": 10.0}], 100.0, 10.0)
    logger.log_model_result(1, [4, 5, 6], [{"epoch": 1, "test_acc": 12.0}], 120.0, 12.0)
    logger.log_data["hardware_stats"].append(
        {
            "timestamp": 0.0,
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "gpu_memory_allocated": 0.0,
            "gpu_memory_reserved": 0.0,
        }
    )
    log_file = logger.finish()
    assert Path(log_file).exists()

    viz = NTKVisualizer(log_file, str(tmp_path))
    viz.generate_all()
    assert (tmp_path / "ntk_correlation_plot.png").exists()
    assert (tmp_path / "training_curves.png").exists()
    assert (tmp_path / "hardware_usage.png").exists()


def test_run_correlation_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FINAL_DATASET", "cifar10", raising=False)
    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr("apply.correlation_experiment.DatasetLoader.get_cifar10", lambda *args, **kwargs: (_dummy_loader(), _dummy_loader()))
    monkeypatch.setattr("apply.correlation_experiment.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.correlation_experiment.NetworkBuilder.build_from_individual", lambda *args, **kwargs: NetworkBuilder.build_from_encoding(make_encoding(unit_num=2, block_nums=[3, 3]), input_channels=3, num_classes=10))
    monkeypatch.setattr("apply.correlation_experiment.NetworkTrainer.train_network", lambda *args, **kwargs: (20.0, [{"epoch": 1, "test_acc": 10.0}, {"epoch": 2, "test_acc": 20.0}]))
    monkeypatch.setattr("apply.correlation_experiment.os.path.dirname", lambda *args, **kwargs: str(tmp_path))
    run_correlation_experiment(num_models=1, full_epochs=2, short_epochs=1)
    assert list(tmp_path.glob("experiment_results/*.json"))


def test_run_ntk_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FINAL_DATASET", "cifar10", raising=False)
    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr("apply.ntk_correlation_experiment.DatasetLoader.get_cifar10", lambda *args, **kwargs: (_dummy_loader(), _dummy_loader()))
    monkeypatch.setattr("apply.ntk_correlation_experiment.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.ntk_correlation_experiment.NTKEvaluator.evaluate_individual", lambda self, ind: setattr(ind, "fitness", 100.0))
    monkeypatch.setattr("apply.ntk_correlation_experiment.NetworkTrainer.train_network", lambda *args, **kwargs: (10.0, [{"epoch": 1, "test_acc": 10.0}]))
    monkeypatch.setattr("apply.ntk_correlation_experiment.os.path.dirname", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr("apply.ntk_correlation_experiment.ExperimentLogger.start_monitoring", lambda *args, **kwargs: None)
    monkeypatch.setattr("apply.ntk_correlation_experiment.ExperimentLogger.stop_monitoring", lambda *args, **kwargs: None)
    run_ntk_experiment(num_models=1, short_epochs=1)
    assert list(tmp_path.glob("ntk_experiment_results/*.json"))
