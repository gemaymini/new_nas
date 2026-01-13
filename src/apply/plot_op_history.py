# -*- coding: utf-8 -*-
"""
Plot operation history counts from JSONL logs.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def load_records(path: str):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records):
    cross_counter = Counter()
    mutation_counter = Counter()
    step_to_fitness = {}

    for rec in records:
        step = rec.get("step")
        ops = rec.get("operations", [])
        fitness = rec.get("fitness")
        if step is not None and fitness is not None:
            step_to_fitness[step] = fitness

        for op in ops:
            if op.get("op") == "uniform_unit_crossover":
                cross_counter["crossover"] += 1
            elif op.get("op") == "copy_parent":
                cross_counter["copy_parent"] += 1
            elif op.get("type") == "mutation":
                for detail in op.get("ops", []):
                    name = detail.get("op", "unknown")
                    if detail.get("applied"):
                        mutation_counter[name] += 1
    return cross_counter, mutation_counter, step_to_fitness


def plot_bars(counter, title, outfile):
    labels, values = zip(*counter.most_common()) if counter else ([], [])
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='steelblue')
    plt.title(title)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile


def plot_fitness(step_to_fitness, outfile):
    if not step_to_fitness:
        return None
    steps = sorted(step_to_fitness.keys())
    values = [step_to_fitness[s] for s in steps]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, values, marker='o', linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Fitness (NTK)")
    plt.title("Fitness over steps")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Plot operation history statistics")
    parser.add_argument("--log_path", type=str, default="logs/op_history.jsonl", help="Path to op_history.jsonl")
    parser.add_argument("--out_dir", type=str, default="logs", help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        raise FileNotFoundError(f"Log file not found: {args.log_path}")
    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.log_path)
    cross_counter, mutation_counter, step_to_fitness = aggregate(records)

    outputs = []
    if cross_counter:
        outputs.append(plot_bars(cross_counter, "Crossover / Copy counts", os.path.join(args.out_dir, "op_cross.png")))
    if mutation_counter:
        outputs.append(plot_bars(mutation_counter, "Mutation counts", os.path.join(args.out_dir, "op_mutation.png")))
    fit_plot = plot_fitness(step_to_fitness, os.path.join(args.out_dir, "op_fitness.png"))
    if fit_plot:
        outputs.append(fit_plot)

    print(f"INFO: saved_plots count={len(outputs)}")
    for p in outputs:
        print(f"INFO: saved_plot={p}")


if __name__ == "__main__":
    main()
