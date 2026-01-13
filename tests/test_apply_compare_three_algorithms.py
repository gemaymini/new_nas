# -*- coding: utf-8 -*-
"""
Tests for apply.compare_three_algorithms.
"""
from types import SimpleNamespace
from pathlib import Path

import torch

from apply.compare_three_algorithms import (
    ModelInfo,
    count_parameters,
    get_model_param_count,
    ThreeStageEA,
    TraditionalEA,
    RandomSearchAlgorithm,
    plot_pareto_comparison,
    save_experiment_results,
)
from core.encoding import Individual
from models.network import NetworkBuilder

from conftest import make_encoding


def test_model_info_to_dict():
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    ind.id = 1
    info = ModelInfo(individual=ind, param_count=1.0, accuracy=90.0, ntk_score=10.0)
    payload = info.to_dict()
    assert payload["id"] == 1
    assert payload["param_count"] == 1.0


def test_count_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    params_m = count_parameters(model)
    assert params_m > 0


def test_get_model_param_count(monkeypatch):
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    monkeypatch.setattr("apply.compare_three_algorithms.NetworkBuilder.build_from_individual", lambda *args, **kwargs: torch.nn.Linear(4, 2))
    assert get_model_param_count(ind) > 0


class DummyEvaluator:
    def evaluate_individual(self, ind, epochs=None):
        return 10.0, {}


def test_three_stage_ea_run(monkeypatch):
    monkeypatch.setattr("apply.compare_three_algorithms.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.compare_three_algorithms.fitness_evaluator.evaluate_individual", lambda ind: setattr(ind, "fitness", 5.0))
    monkeypatch.setattr("apply.compare_three_algorithms.clear_gpu_memory", lambda *args, **kwargs: None)
    ea = ThreeStageEA(max_evaluations=2, population_size=1, top_n1=1, top_n2=1, short_epochs=1, full_epochs=1)
    models = ea.run(evaluator=DummyEvaluator())
    assert len(models) == 1


def test_traditional_ea_run(monkeypatch):
    monkeypatch.setattr("apply.compare_three_algorithms.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.compare_three_algorithms.clear_gpu_memory", lambda *args, **kwargs: None)
    ea = TraditionalEA(max_evaluations=2, population_size=1, top_n=1, search_epochs=1, full_epochs=1)
    models = ea.run(evaluator=DummyEvaluator())
    assert len(models) == 1


def test_random_search_run(monkeypatch):
    monkeypatch.setattr("apply.compare_three_algorithms.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.compare_three_algorithms.clear_gpu_memory", lambda *args, **kwargs: None)
    rs = RandomSearchAlgorithm(num_samples=1, full_epochs=1)
    models = rs.run(evaluator=DummyEvaluator())
    assert len(models) == 1


def test_plot_and_save_results(tmp_path):
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    ind.id = 1
    model = ModelInfo(individual=ind, param_count=1.0, accuracy=90.0, ntk_score=10.0)
    plot_pareto_comparison([model], [model], [model], output_dir=str(tmp_path), show_plot=False)
    out = save_experiment_results([model], [model], [model], output_dir=str(tmp_path), config_dict={"x": 1})
    assert Path(out).exists()
