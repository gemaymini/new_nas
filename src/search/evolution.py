# -*- coding: utf-8 -*-
"""
Aging Evolution (Regularized Evolution) algorithm implementation.
"""
import random
import os
import pickle
import time
import threading
import json
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Optional

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, FinalEvaluator
from utils.generation import generate_valid_child
from utils.constraints import evaluate_encoding_params
from utils.logger import logger, tb_logger


class AgingEvolutionNAS:
    def __init__(self):
        self.population_size = config.POPULATION_SIZE
        self.max_gen = config.MAX_GEN  # Total individuals to evaluate

        # Population & history state.
        self.population = deque()
        self.history: List[Individual] = []
        self.lock = threading.Lock()

        # Track unique encodings to skip duplicates.
        self.seen_encodings: set = set()
        self.duplicate_count = 0

        # NTK history: [(step, individual_id, ntk_value, encoding), ...]
        self.ntk_history: List[Tuple[int, int, float, list]] = []

        self.start_time = time.time()

        # Timing stats for phases.
        self.search_time = 0.0
        self.short_train_time = 0.0
        self.full_train_time = 0.0
        self.time_stats: dict = {}

        self._log_search_space_info()

    def _log_search_space_info(self):
        logger.info(config.get_search_space_summary())
        logger.info(f"Aging Evolution Config: Pop Size={self.population_size}, Total Gen={self.max_gen}")

    def _is_duplicate(self, encoding: List[int]) -> bool:
        """Check whether this encoding has been seen."""
        enc_tuple = tuple(encoding)
        return enc_tuple in self.seen_encodings

    def _register_encoding(self, encoding: List[int]):
        """Register encoding as seen."""
        enc_tuple = tuple(encoding)
        self.seen_encodings.add(enc_tuple)

    def initialize_population(self):
        """Initialize the population with random individuals."""
        logger.info("Initializing population...")

        while len(self.population) < self.population_size:
            ind = population_initializer.create_valid_individual()
            if self._is_duplicate(ind.encoding):
                self.duplicate_count += 1
                print("WARN: duplicate architecture, resampling")
                continue

            self._register_encoding(ind.encoding)
            ind.id = len(self.population)
            fitness_evaluator.evaluate_individual(ind)
            self.population.append(ind)
            self.history.append(ind)

            step = 0  # Initialization step
            self.ntk_history.append((step, ind.id, ind.fitness, ind.encoding.copy()))

            if len(self.population) % 10 == 0:
                logger.info(
                    f"Initialized {len(self.population)}/{self.population_size} individuals "
                    f"(duplicates skipped: {self.duplicate_count})"
                )

        logger.info(
            f"Population initialized. Size: {len(self.population)}, Duplicates skipped: {self.duplicate_count}"
        )
        self._record_statistics()
        self._save_ntk_history()

    def _select_parents(self) -> Tuple[Individual, Individual]:
        """Tournament selection to choose two parents."""
        current_pop_list = list(self.population)
        parents = selection_operator.tournament_selection(
            current_pop_list,
            tournament_size=config.TOURNAMENT_SIZE,
            num_winners=config.TOURNAMENT_WINNERS,
        )
        return parents[0], parents[1]

    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        """Generate one offspring using crossover and mutation."""
        return generate_valid_child(
            parent1=parent1,
            parent2=parent2,
            crossover_fn=crossover_operator.crossover,
            mutation_fn=mutation_operator.mutate,
            repair_fn=self._repair_individual,
            resample_fn=population_initializer.create_valid_individual,
            crossover_prob=config.PROB_CROSSOVER,
            mutation_prob=config.PROB_MUTATION,
            max_attempts=50,
        )

    def _repair_individual(self, ind: Individual) -> Individual:
        while True:
            ind = mutation_operator.mutate(ind)
            if Encoder.validate_encoding(ind.encoding):
                return ind
            print("WARN: repair failed, mutating again")

    def step(self) -> bool:
        """
        One step of aging evolution:
        1) select parents
        2) generate child (skip if duplicate)
        3) evaluate child
        4) update population FIFO
        """
        child = None
        while True:
            parent1, parent2 = self._select_parents()
            child = self._generate_offspring(parent1, parent2)

            if self._is_duplicate(child.encoding):
                self.duplicate_count += 1
                print("WARN: duplicate architecture, resampling")
                continue

            # Enforce param bounds before expensive eval/logging; resample instead of recording failure.
            ok, reason, param_count = evaluate_encoding_params(child.encoding)
            if not ok:
                logger.warning(f"Resampling child: param bounds failed ({reason})")
                continue
            child.param_count = param_count
            break

        self._register_encoding(child.encoding)
        child.id = len(self.history)

        fitness_evaluator.evaluate_individual(child)

        current_step = len(self.history) - len(self.population) + 1

        logger.log_operation(
            {
                "step": current_step,
                "child_id": child.id,
                "parent_ids": [parent1.id, parent2.id],
                "fitness": child.fitness,
                "operations": child.op_history,
                "encoding": child.encoding,
            }
        )

        self.ntk_history.append((current_step, child.id, child.fitness, child.encoding.copy()))

        with self.lock:
            self.population.popleft()
            self.population.append(child)
            self.history.append(child)

        if len(self.history) % 10 == 0:
            logger.info(
                f"Step {len(self.history)-len(self.population)}/{self.max_gen}: "
                f"Child Fitness={child.fitness:.4f} (duplicates skipped: {self.duplicate_count})"
            )
            self._record_statistics()
            self._save_ntk_history()

        return True

    def run_search(self):
        """Main loop for aging evolution search."""
        logger.info(f"Starting Aging Evolution Search for {self.max_gen} steps...")
        search_start_time = time.time()

        if not self.population:
            self.initialize_population()

        while len(self.history) - len(self.population) < self.max_gen:
            self.step()
            if (len(self.history) - len(self.population)) % 100 == 0:
                self.save_checkpoint()

        self.search_time = time.time() - search_start_time
        logger.info(f"Search completed. Search time: {self._format_time(self.search_time)}")
        logger.info(
            f"Search statistics: {len(self.history)} valid individuals evaluated, "
            f"{self.duplicate_count} duplicates skipped, {len(self.seen_encodings)} unique architectures"
        )
        self.save_checkpoint()
        self._save_ntk_history()
        self.plot_ntk_curve()

    def run_screening_and_training(self):
        """Multi-stage screening and final training."""
        logger.info("Starting Screening and Training Phase...")

        unique_history = {}
        for ind in self.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            else:
                if ind.fitness is not None and (
                    unique_history[enc_tuple].fitness is None
                    or ind.fitness < unique_history[enc_tuple].fitness
                ):
                    unique_history[enc_tuple] = ind

        candidates = list(unique_history.values())
        candidates.sort(
            key=lambda x: x.fitness if x.fitness is not None else float("inf"),
            reverse=False,
        )

        if not candidates:
            logger.warning("No evaluated individuals available for screening; skipping training stage.")
            return None

        top_n1 = candidates[:config.HISTORY_TOP_N1]
        logger.info(
            f"Selected Top {config.HISTORY_TOP_N1} candidates from {len(candidates)} "
            "unique history individuals based on NTK."
        )

        logger.info(
            f"Starting Short Training ({config.SHORT_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N1}..."
        )
        short_train_start_time = time.time()

        evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)

        short_results = []
        for i, ind in enumerate(top_n1):
            logger.info(f"Short Train [{i+1}/{len(top_n1)}] ID: {ind.id}")
            acc, _ = evaluator.evaluate_individual(ind, epochs=config.SHORT_TRAIN_EPOCHS)
            ind.quick_score = acc
            short_results.append(ind)

        self.short_train_time = time.time() - short_train_start_time
        logger.info(f"Short Training completed. Time: {self._format_time(self.short_train_time)}")

        short_results.sort(
            key=lambda x: x.quick_score if x.quick_score else float("-inf"),
            reverse=True,
        )
        top_n2 = short_results[:config.HISTORY_TOP_N2]
        logger.info(
            f"Selected Top {config.HISTORY_TOP_N2} candidates based on Short Training Accuracy."
        )

        if not top_n2:
            logger.warning("No candidates advanced to full training; skipping final stage.")
            self._save_time_stats()
            return None

        logger.info(
            f"Starting Full Training ({config.FULL_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N2}..."
        )
        full_train_start_time = time.time()

        final_results = []
        best_final_ind = None
        best_final_acc = float("-inf")

        for i, ind in enumerate(top_n2):
            logger.info(f"Full Train [{i+1}/{len(top_n2)}] ID: {ind.id}")
            acc, result = evaluator.evaluate_individual(ind, epochs=config.FULL_TRAIN_EPOCHS)

            logger.info(f"Individual {ind.id} Final Accuracy: {acc:.2f}%")

            if acc > best_final_acc:
                best_final_acc = acc
                best_final_ind = ind

            final_results.append(result)

        self.full_train_time = time.time() - full_train_start_time
        logger.info(f"Full Training completed. Time: {self._format_time(self.full_train_time)}")

        self._save_time_stats()

        if best_final_ind is None:
            logger.warning("No valid final model produced during full training.")
            return None

        logger.info(f"Best Final Model: ID={best_final_ind.id}, Acc={best_final_acc:.2f}%")
        return best_final_ind

    def _record_statistics(self):
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        valid_fitnesses = [f for f in fitnesses if f < 100000.0]

        if valid_fitnesses:
            avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
            best_fitness = min(valid_fitnesses)
        elif fitnesses:
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = min(fitnesses)
        else:
            avg_fitness = float("inf")
            best_fitness = float("inf")

        stats = {
            "generation": len(self.history) - len(self.population),
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "population_size": len(self.population),
        }

        logger.log_generation(
            len(self.history) - len(self.population),
            best_fitness,
            avg_fitness,
            len(self.population),
        )
        tb_logger.log_generation_stats(len(self.history) - len(self.population), stats)

        unit_counts = {}
        for ind in self.population:
            if ind.encoding:
                unit_num = ind.encoding[0]
                unit_counts[unit_num] = unit_counts.get(unit_num, 0) + 1
        logger.log_unit_stats(len(self.history) - len(self.population), unit_counts)

    def save_checkpoint(self, filepath: str = None):
        if filepath is None:
            if not os.path.exists(config.CHECKPOINT_DIR):
                os.makedirs(config.CHECKPOINT_DIR)
            filepath = os.path.join(
                config.CHECKPOINT_DIR,
                f"checkpoint_step{len(self.history)-len(self.population)}.pkl",
            )

        checkpoint = {
            "population": list(self.population),
            "history": self.history,
            "ntk_history": self.ntk_history,
            "seen_encodings": self.seen_encodings,
            "duplicate_count": self.duplicate_count,
            "search_time": self.search_time,
            "short_train_time": self.short_train_time,
            "full_train_time": self.full_train_time,
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
        self.population = deque(checkpoint["population"])
        self.history = checkpoint["history"]
        self.ntk_history = checkpoint.get("ntk_history", [])
        self.seen_encodings = checkpoint.get("seen_encodings", set())
        self.duplicate_count = checkpoint.get("duplicate_count", 0)
        self.search_time = checkpoint.get("search_time", 0.0)
        self.short_train_time = checkpoint.get("short_train_time", 0.0)
        self.full_train_time = checkpoint.get("full_train_time", 0.0)
        logger.info(
            f"Checkpoint loaded from {filepath}, duplicates skipped so far: {self.duplicate_count}"
        )

    def _save_ntk_history(self, filepath: str = None):
        """Persist NTK history to JSON."""
        if not self.ntk_history:
            return

        if filepath is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            filepath = os.path.join(config.LOG_DIR, "ntk_history.json")

        data = []
        for step, ind_id, ntk_value, encoding in self.ntk_history:
            data.append(
                {
                    "step": step,
                    "individual_id": ind_id,
                    "ntk": ntk_value if ntk_value is not None else None,
                    "encoding": encoding,
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"NTK history saved to {filepath}")

    def plot_ntk_curve(self, output_path: str = None):
        """Plot NTK history curves."""
        if not self.ntk_history:
            logger.warning("No NTK history to plot!")
            return

        if output_path is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            output_path = os.path.join(config.LOG_DIR, "ntk_curve.png")

        steps = []
        ntk_values = []
        for step, ind_id, ntk, encoding in self.ntk_history:
            if ntk is not None and ntk < 100000:
                steps.append(step)
                ntk_values.append(ntk)

        if not steps:
            logger.warning("No valid NTK values to plot!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.scatter(steps, ntk_values, alpha=0.3, s=10, c="blue")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("NTK Condition Number")
        ax1.set_title("All Individuals NTK Values")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        window_size = max(10, len(ntk_values) // 50)
        if len(ntk_values) >= window_size:
            moving_avg = []
            for i in range(len(ntk_values) - window_size + 1):
                avg = sum(ntk_values[i:i + window_size]) / window_size
                moving_avg.append(avg)
            moving_avg_steps = steps[window_size - 1:]
            ax2.plot(
                moving_avg_steps,
                moving_avg,
                "r-",
                linewidth=2,
                label=f"Moving Avg (window={window_size})",
            )
            ax2.scatter(steps, ntk_values, alpha=0.2, s=5, c="blue", label="Individual NTK")
            ax2.legend()
        else:
            ax2.scatter(steps, ntk_values, alpha=0.5, s=10, c="blue")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("NTK Condition Number")
        ax2.set_title("NTK with Moving Average")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        step_best = {}
        for step, ind_id, ntk, encoding in self.ntk_history:
            if ntk is not None and ntk < 100000:
                if step not in step_best or ntk < step_best[step]:
                    step_best[step] = ntk

        sorted_steps = sorted(step_best.keys())
        best_ntks = [step_best[s] for s in sorted_steps]

        cumulative_best = []
        current_best = float("inf")
        for ntk in best_ntks:
            current_best = min(current_best, ntk)
            cumulative_best.append(current_best)

        ax3.plot(sorted_steps, best_ntks, "g-", alpha=0.5, label="Best per Step")
        ax3.plot(sorted_steps, cumulative_best, "r-", linewidth=2, label="Cumulative Best")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("NTK Condition Number")
        ax3.set_title("Best NTK Progress")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.hist(ntk_values, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax4.axvline(
            min(ntk_values),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Best: {min(ntk_values):.2f}",
        )
        ax4.axvline(
            sum(ntk_values) / len(ntk_values),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {sum(ntk_values)/len(ntk_values):.2f}",
        )
        ax4.set_xlabel("NTK Condition Number")
        ax4.set_ylabel("Count")
        ax4.set_title("NTK Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"NTK curve saved to {output_path}")
        logger.info(
            f"NTK Statistics: Total={len(ntk_values)}, Best={min(ntk_values):.4f}, "
            f"Mean={sum(ntk_values)/len(ntk_values):.4f}, Worst={max(ntk_values):.4f}"
        )

    def _format_time(self, seconds: float) -> str:
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}min ({seconds:.0f}s)"
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.2f}h ({minutes:.0f}min)"

    def _save_time_stats(self, filepath: str = None):
        """Save timing stats to JSON and log summary."""
        total_time = self.search_time + self.short_train_time + self.full_train_time

        self.time_stats = {
            "search_phase": {
                "time_seconds": self.search_time,
                "time_formatted": self._format_time(self.search_time),
                "description": f"Search phase (NTK eval {self.max_gen} individuals)",
            },
            "short_training_phase": {
                "time_seconds": self.short_train_time,
                "time_formatted": self._format_time(self.short_train_time),
                "description": (
                    f"Short training (Top {config.HISTORY_TOP_N1} models, "
                    f"{config.SHORT_TRAIN_EPOCHS} epochs)"
                ),
            },
            "full_training_phase": {
                "time_seconds": self.full_train_time,
                "time_formatted": self._format_time(self.full_train_time),
                "description": (
                    f"Full training (Top {config.HISTORY_TOP_N2} models, "
                    f"{config.FULL_TRAIN_EPOCHS} epochs)"
                ),
            },
            "total": {
                "time_seconds": total_time,
                "time_formatted": self._format_time(total_time),
                "description": "Total runtime",
            },
        }

        if filepath is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            filepath = os.path.join(config.LOG_DIR, "time_stats.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.time_stats, f, indent=2, ensure_ascii=False)

        logger.info("=" * 60)
        logger.info("Timing Summary")
        logger.info("=" * 60)
        logger.info(f"Search phase:     {self._format_time(self.search_time)}")
        logger.info(f"Short training:   {self._format_time(self.short_train_time)}")
        logger.info(f"Full training:    {self._format_time(self.full_train_time)}")
        logger.info("-" * 60)
        logger.info(f"Total:            {self._format_time(total_time)}")
        logger.info("=" * 60)
        logger.info(f"Time stats saved to {filepath}")
