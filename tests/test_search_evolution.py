# -*- coding: utf-8 -*-
"""
Tests for search.evolution.
"""
import json
from pathlib import Path

from core.encoding import Individual
from search.evolution import AgingEvolutionNAS
from utils.logger import logger

from conftest import make_encoding


def _make_individual(idx, encoding):
    ind = Individual(encoding)
    ind.id = idx
    ind.fitness = 1.0 + idx
    return ind


def test_format_time():
    nas = AgingEvolutionNAS()
    assert nas._format_time(10.0).endswith("s")
    assert "min" in nas._format_time(120.0)
    assert "h" in nas._format_time(3600.0)


def test_initialize_population(monkeypatch):
    encodings = [
        make_encoding(unit_num=2, block_nums=[3, 3], concat_last=False),
        make_encoding(unit_num=2, block_nums=[3, 3], concat_last=True),
    ]
    calls = {"idx": 0}

    def mock_create_valid_individual():
        idx = calls["idx"]
        calls["idx"] += 1
        return _make_individual(idx, encodings[idx % len(encodings)])

    monkeypatch.setattr("search.evolution.population_initializer.create_valid_individual", mock_create_valid_individual)
    monkeypatch.setattr("search.evolution.fitness_evaluator.evaluate_individual", lambda ind: setattr(ind, "fitness", 1.0))
    monkeypatch.setattr(logger, "log_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr("search.evolution.AgingEvolutionNAS._save_ntk_history", lambda *args, **kwargs: None)
    nas = AgingEvolutionNAS()
    nas.population_size = 2
    nas.initialize_population()
    assert len(nas.population) == 2
    assert len(nas.history) == 2
    assert len(nas.ntk_history) == 2


def test_step_updates_population(monkeypatch):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    nas = AgingEvolutionNAS()
    nas.population_size = 2
    nas.population.append(_make_individual(0, encoding))
    nas.population.append(_make_individual(1, encoding))
    nas.history = list(nas.population)

    monkeypatch.setattr("search.evolution.fitness_evaluator.evaluate_individual", lambda ind: setattr(ind, "fitness", 1.0))
    monkeypatch.setattr(logger, "log_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(nas, "_select_parents", lambda: (nas.history[0], nas.history[1]))
    monkeypatch.setattr(nas, "_generate_offspring", lambda p1, p2: _make_individual(2, make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr(nas, "_is_duplicate", lambda enc: False)

    nas.step()
    assert len(nas.population) == 2
    assert len(nas.history) == 3


def test_save_and_load_checkpoint(tmp_path, monkeypatch):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    nas = AgingEvolutionNAS()
    nas.population.append(_make_individual(0, encoding))
    nas.history.append(_make_individual(0, encoding))
    path = tmp_path / "checkpoint.pkl"
    nas.save_checkpoint(filepath=str(path))
    assert path.exists()

    nas2 = AgingEvolutionNAS()
    nas2.load_checkpoint(str(path))
    assert len(nas2.population) == 1
    assert len(nas2.history) == 1


def test_save_ntk_history(tmp_path):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    nas = AgingEvolutionNAS()
    nas.ntk_history = [(0, 0, 1.23, encoding)]
    path = tmp_path / "ntk_history.json"
    nas._save_ntk_history(filepath=str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["ntk"] == 1.23


def test_save_time_stats(tmp_path):
    nas = AgingEvolutionNAS()
    nas.search_time = 1.0
    nas.short_train_time = 2.0
    nas.full_train_time = 3.0
    path = tmp_path / "time_stats.json"
    nas._save_time_stats(filepath=str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "total" in data


def test_plot_ntk_curve(tmp_path, monkeypatch):
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    nas = AgingEvolutionNAS()
    nas.ntk_history = [(0, 0, 1.0, encoding), (1, 1, 2.0, encoding)]
    out = tmp_path / "ntk_curve.png"
    nas.plot_ntk_curve(output_path=str(out))
    assert out.exists()
