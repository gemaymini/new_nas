# -*- coding: utf-8 -*-
"""
Tests for plotting utilities in apply.
"""
import json
from pathlib import Path

from apply.plot_algorithm_comparison import (
    generate_simulated_data,
    merge_experiment_results,
    plot_algorithm_comparison,
    print_statistics,
)
from apply.plot_compare_evolution_vs_random import load_data as load_compare_data, plot_comparison as plot_compare
from apply.plot_ntk_curve import load_ntk_history_from_json, plot_ntk_curve
from apply.plot_ntk_vs_shortacc import load_and_process_data, generate_virtual_points
from apply.plot_short_vs_full import load_experiment_logs, plot_correlation
from apply.plot_op_history import load_records, aggregate, plot_bars, plot_fitness


def test_plot_algorithm_comparison(tmp_path, capsys):
    data = generate_simulated_data()
    print_statistics(data)
    out = tmp_path / "algo.png"
    plot_algorithm_comparison(data, output_path=str(out), show_plot=False, show_inset=False)
    assert (tmp_path / "algo.pdf").exists()


def test_plot_compare_evolution_vs_random(tmp_path):
    payload = {
        "evolution_curve": [100.0, 90.0, 80.0],
        "random_curve": [120.0, 110.0, 100.0],
        "evolution_all_ntk": [100.0, 90.0, 80.0],
        "random_all_ntk": [120.0, 110.0, 100.0],
    }
    json_path = tmp_path / "comp.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    evo, rand, evo_all, rand_all = load_compare_data(str(json_path))
    out = tmp_path / "comp.png"
    plot_compare(evo, rand, evo_all, rand_all, str(out))
    assert (tmp_path / "comp.pdf").exists()


def test_plot_ntk_curve_from_json(tmp_path):
    payload = [{"step": 0, "individual_id": 1, "ntk": 1.0, "encoding": [1, 2, 3]}]
    json_path = tmp_path / "ntk.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    ntk_history = load_ntk_history_from_json(str(json_path))
    out = tmp_path / "ntk_curve.png"
    plot_ntk_curve(ntk_history, output_path=str(out), title_prefix="Test ")
    assert out.exists()


def test_plot_ntk_vs_shortacc(tmp_path, monkeypatch):
    log = {
        "models": [
            {"ntk_cond": 10.0, "encoding": [1, 2, 3], "history": [{"epoch": 15, "test_acc": 50.0}]},
            {"ntk_cond": 12.0, "encoding": [4, 5, 6], "history": [{"epoch": 15, "test_acc": 55.0}]},
        ]
    }
    log_path = tmp_path / "ntk_experiment_log_000.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    monkeypatch.setattr("apply.plot_ntk_vs_shortacc.RESULTS_DIR", str(tmp_path))
    data = load_and_process_data()
    assert len(data) == 2
    x_real = [1.0, 2.0]
    y_real = [50.0, 55.0]
    x_gen, y_gen = generate_virtual_points(x_real, y_real, target_total=2)
    assert len(x_gen) == 0


def test_plot_short_vs_full(tmp_path, monkeypatch):
    log = {
        "meta": {"config": {"short_epochs": 1, "full_epochs": 2, "dataset": "cifar10"}},
        "models": [
            {"short_acc": 50.0, "full_acc": 60.0},
            {"short_acc": 55.0, "full_acc": 65.0},
        ],
    }
    log_path = tmp_path / "experiment_log_000.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    all_data, meta = load_experiment_logs(log_files=[str(log_path)])
    out = tmp_path / "short_full"
    monkeypatch.setattr("apply.plot_short_vs_full.ENABLE_VIRTUAL_POINTS", False)
    plot_correlation(all_data, meta, output_path=str(out))
    assert (tmp_path / "short_full.pdf").exists()


def test_plot_op_history(tmp_path):
    log_path = tmp_path / "op_history.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "fitness": 10.0,
                        "operations": [
                            {"op": "uniform_unit_crossover"},
                            {"op": "copy_parent"},
                            {"type": "mutation", "ops": [{"op": "add_block", "applied": True}]},
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    records = load_records(str(log_path))
    cross_counter, mut_counter, step_to_fitness = aggregate(records)
    out1 = plot_bars(cross_counter, "Cross", str(tmp_path / "cross.png"))
    out2 = plot_bars(mut_counter, "Mut", str(tmp_path / "mut.png"))
    out3 = plot_fitness(step_to_fitness, str(tmp_path / "fitness.png"))
    assert Path(out1).exists()
    assert Path(out2).exists()
    assert Path(out3).exists()
