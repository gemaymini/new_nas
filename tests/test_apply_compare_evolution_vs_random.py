# -*- coding: utf-8 -*-
"""
Tests for apply.compare_evolution_vs_random.
"""
from apply.compare_evolution_vs_random import RandomSearch, AgingEvolutionSearch, plot_comparison
from core.encoding import Individual

from conftest import make_encoding


def test_random_search_run(monkeypatch):
    monkeypatch.setattr("apply.compare_evolution_vs_random.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.compare_evolution_vs_random.fitness_evaluator.evaluate_individual", lambda ind: setattr(ind, "fitness", 10.0))
    rs = RandomSearch(max_evaluations=2)
    history, curve, all_vals = rs.run()
    assert len(history) == 2
    assert len(curve) == 2
    assert len(all_vals) == 2


def test_aging_evolution_run(monkeypatch):
    monkeypatch.setattr("apply.compare_evolution_vs_random.population_initializer.create_valid_individual", lambda: Individual(make_encoding(unit_num=2, block_nums=[3, 3])))
    monkeypatch.setattr("apply.compare_evolution_vs_random.fitness_evaluator.evaluate_individual", lambda ind: setattr(ind, "fitness", 10.0))
    ae = AgingEvolutionSearch(max_evaluations=2, population_size=1)
    history, curve, all_vals = ae.run()
    assert len(history) == 2
    assert len(curve) == 2
    assert len(all_vals) == 2
