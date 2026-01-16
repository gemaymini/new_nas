# Neural Architecture Search (NAS) with Aging Evolution and NTK Screening

This project implements an evolutionary NAS framework that screens candidate architectures using Neural Tangent Kernel (NTK) condition numbers, followed by staged training to identify high-performing models.

## Features
- **Aging Evolution search** with crossover/mutation and duplicate filtering.
- **NTK-based zero-cost proxy** to quickly score architectures under parameter constraints.
- **Two-stage training**: short screening then full training of top candidates.
- **Extensible search space** with variable-length encodings and channel/stride/skip options.
- **Experiment utilities**: plotting, comparisons vs. random search, correlation studies.
- **Test suite with dependency stubs** to keep CI lightweight.

## Repository Structure
- `src/main.py` — CLI entry for running NAS search/training.
- `src/configuration/config.py` — All hyperparameters and global settings.
- `src/core/` — Encoding utilities and search-space sampling.
- `src/search/` — Evolutionary logic (aging evolution, mutation, crossover).
- `src/engine/` — Evaluators (NTK + final training) and trainer.
- `src/models/` — Network builder for searched architectures.
- `src/utils/` — Logging, constraints, offspring generation helpers.
- `src/apply/` — Experiment and plotting scripts.
- `tests/` — Pytest suite with lightweight stubs for heavy deps.

## Installation
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Note: Large binaries (torch, torchvision, numpy) must be compatible with your Python version/GPU. Ensure CUDA availability if using GPU; the code falls back to CPU when needed.

## Quick Start
Run an end-to-end NAS search on CIFAR-10:
```bash
python src/main.py --dataset cifar10
```
Key optional flags (see `parse_args` in `src/main.py`):
- `--optimizer {adamw,sgd}`
- `--dataset {cifar10,cifar100,imagenet}`
- `--imagenet_root PATH` (required when dataset=imagenet)
- `--seed INT`
- `--resume CHECKPOINT_PATH` (resume a saved search)
- `--no_final_eval` (skip final training if you integrate such toggle)

## Configuration Highlights (`configuration.config.Config`)
- Search: population size, max generations, tournament sizes, crossover/mutation probabilities.
- Search space: unit/block count ranges, channels/groups/pooling/SENet/activations/skip types/kernel sizes/expansion.
- Constraints: min/max parameter counts, dataset-specific bounds.
- Training: batch sizes, optimizer defaults, warmup + cosine scheduler, early stopping.
- Logging/IO: log/checkpoint/tensorboard directories, saving failed individuals.

## How It Works
1) **Population initialization**: `core.search_space.population_initializer` samples valid encodings within constraints.
2) **Fitness (NTK)**: `engine.evaluator.NTKEvaluator` builds a network, checks parameter bounds, computes NTK condition number (lower is better). Invalid or oversized models are penalized.
3) **Evolution loop**: `search.evolution.AgingEvolutionNAS` selects parents (tournament), generates offspring (`utils.generation.generate_valid_child` with crossover/mutation), evaluates NTK, and maintains a FIFO population.
4) **Screening & training**: Top NTK models undergo short training; best few get full training via `engine.evaluator.FinalEvaluator` and `engine.trainer.NetworkTrainer`.
5) **Artifacts**: Checkpoints, NTK history JSON/plots, trained models, and logs are written under configured directories.

## Running Experiments/Plots
All scripts live in `src/apply/` and are runnable as stand-alone modules, e.g.:
```bash
python src/apply/compare_evolution_vs_random.py --max_eval 50 --pop_size 10
python src/apply/plot_ntk_curve.py --input logs/ntk_history.json
python src/apply/ntk_correlation_experiment.py
```
These scripts use Agg backend by default; they should run headless.

## Extending the Search Space
1) Add options/constants to `Config` (e.g., new activation or kernel sizes).
2) Extend `BlockParams`/`BLOCK_PARAM_COUNT` and sampling logic in `core.search_space`.
3) Update validation in `core.encoding.Encoder.validate_encoding`.
4) Adjust model construction in `models.network` to honor new params.

## Notes on Devices and Memory
- Always guard CUDA usage with `torch.cuda.is_available()`; code falls back to CPU.
- After heavy evaluation/training, call `engine.evaluator.clear_gpu_memory()` as needed.
- Keep parameter counts within configured bounds to avoid OOM and invalid encodings.

## Logging & Outputs
- Logs: `logs/` (timestamped files via `utils.logger`).
- TensorBoard: `runs/` (if enabled).
- Checkpoints: `checkpoints/` (search checkpoints and trained models).
- NTK history: `logs/ntk_history.json` and plot PNG via `AgingEvolutionNAS.plot_ntk_curve`.

## Safety and robustness
- Handle missing datasets/checkpoints with clear errors (ImageNet path must exist).
- Avoid implicit downloads in library calls where possible.
- Keep CLI defaults backward-compatible; tolerate missing optional args in scripts.
