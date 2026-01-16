# Copilot Instructions for this NAS Project

Use these guidelines when generating code or suggestions in this repository to stay aligned with existing patterns, constraints, and tests.

## Project overview
- Domain: evolutionary neural architecture search with NTK-based screening and staged training.
- Entry point: `src/main.py` (CLI). Core search: `src/search/evolution.py`, `src/search/mutation.py`. Evaluation/training: `src/engine/evaluator.py`, `src/engine/trainer.py`. Model building: `src/models/network.py`. Configuration: `src/configuration/config.py`. Utilities: `src/utils/`. Experiments/plots: `src/apply/`. Tests: `tests/` (heavy dependencies are stubbed there).

## Coding conventions
- Python 3.10+. Prefer explicit imports from `src` (avoid fragile relative imports). Keep functions cohesive and short; add type hints where they improve clarity.
- Logging: use `utils.logger.logger` (info/debug/warning/error). Reserve `print` for simple CLI status in scripts.
- Comments: only for non-obvious logic (shape assumptions, probabilistic choices, corner-case guards). Avoid restating code.
- Configuration: add/adjust knobs in `Config` (`src/configuration/config.py`); do not scatter magic numbers.

## Architecture and constraints
- Always validate encodings with `core.encoding.Encoder.validate_encoding`; keep concat skips (`skip_type == 1`) only on the last block of each unit.
- Respect parameter-count limits via `utils.constraints` and `Config`; never hardcode bounds or bypass checks.
- When modifying search/mutation/generation, ensure individuals remain valid and non-duplicate (reuse helpers like `utils.generation.generate_valid_child`).

## Devices, performance, and memory
- Guard CUDA usage with `torch.cuda.is_available()` and provide CPU fallback. Avoid implicit `.cuda()` calls.
- Free GPU/CPU memory after heavy ops (e.g., `clear_gpu_memory()`); avoid holding unnecessary tensors.
- Plots/scripts should run headless: set Agg backend when needed and avoid `plt.show()` by default.

## Testing and stubs
- Tests rely on lightweight stubs in `tests/conftest.py` for numpy/torchvision/pandas/scipy/PIL. Keep new dependencies optional or stubbed similarly to avoid breaking tests.
- When adding CLI args, keep backwards compatibility with tests that may omit fields; update `main.py` defaults accordingly.
- Preserve deterministic behavior in tests (respect fixed seeds in fixtures).

## File organization and additions
- Core search/eval code belongs in `src/search` or `src/engine`; shared helpers in `src/utils`; model blocks in `src/models`.
- Plotting/experiment utilities belong in `src/apply` and should be runnable as scripts.
- Prefer extending existing helpers over duplicating logic; keep new helpers small and close to their usage.

## Safety and resilience
- Handle missing files/dirs gracefully (checkpoints, datasets) with clear error messages.
- Avoid assuming network access or GPU presence in default/test code paths.

## CLI behavior
- `main.py` should tolerate missing optional args (sensible defaults; no crashes when fields are absent).
- Scripts under `src/apply/` should parse args with defaults and avoid side effects on import.

## Non-goals
- Do not introduce training-time downloads or heavy, un-stubbed dependencies.
- Do not add verbose boilerplate comments or redundant logging.
