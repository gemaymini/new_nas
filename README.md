# Neural Architecture Search (NAS) Project

This project implements an Evolutionary Neural Architecture Search (NAS) algorithm.

## Project Structure

- `new_nas/`
  - `core/`: Core components (encoding, search space).
  - `search/`: Search algorithms (Evolution, NSGA-II, Mutation).
  - `model/`: Neural network definitions.
  - `engine/`: Training and evaluation logic.
  - `data/`: Dataset loading.
  - `utils/`: Configuration and logging.
  - `main.py`: Entry point.

## Installation

```bash
pip install torch torchvision numpy
```

## Usage

### Basic Run
```bash
python -m new_nas.main
```

### Test Mode
```bash
python -m new_nas.main --test
```

### Custom Parameters
```bash
python -m new_nas.main --population_size 20 --max_gen 50
```

## Features

- **Modular Design**: Clean separation of concerns.
- **Evolutionary Algorithm**: 3-stage evolution with NSGA-II.
- **NTK Evaluation**: Fast proxy evaluation.
- **Distributed Ready**: Modular design allows for easy distributed extension.
