# -*- coding: utf-8 -*-
"""
Shared pytest fixtures and helpers.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import importlib.util
import types

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from configuration.config import config
from core.encoding import BlockParams, Encoder
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")


def _install_scipy_stubs():
    """
    Install minimal stub implementations for selected ``scipy`` APIs when
    the real ``scipy`` package is not installed.

    The test suite treats ``scipy`` as an optional dependency. To allow tests
    to run in environments where ``scipy`` is unavailable, this helper
    registers lightweight stand-ins for ``scipy``, ``scipy.stats``, and
    ``scipy.spatial`` in ``sys.modules``. When a real ``scipy`` installation
    is present, this function is a no-op and the genuine package is used.
    """
    if importlib.util.find_spec("scipy") is not None:
        return

    class _DummyNorm:
        @staticmethod
        def pdf(x, mu=0.0, sigma=1.0):
            return np.zeros_like(np.asarray(x, dtype=float))

    def _corr_stub(x, y):
        return 0.0, 1.0

    class _DummyHull:
        def __init__(self, points):
            self.vertices = list(range(len(points)))

    stats_mod = types.SimpleNamespace(norm=_DummyNorm, pearsonr=_corr_stub, spearmanr=_corr_stub)
    spatial_mod = types.SimpleNamespace(ConvexHull=_DummyHull)
    scipy_mod = types.SimpleNamespace(stats=stats_mod, spatial=spatial_mod)

    import sys
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.stats"] = stats_mod
    sys.modules["scipy.spatial"] = spatial_mod


_install_scipy_stubs()


def _install_pandas_stubs():
    if importlib.util.find_spec("pandas") is not None:
        return

    class _DummySeries:
        def __init__(self, values):
            self.values = np.asarray(values)

        @property
        def iloc(self):
            return self

        def __getitem__(self, idx):
            return self.values[idx]

        def replace(self, old, new):
            arr = self.values.copy()
            arr[arr == old] = new
            return arr

        def __sub__(self, other):
            return self.values - other

    class _DummyDataFrame:
        def __init__(self, data):
            self._data = list(data)

        def __getitem__(self, key):
            return _DummySeries([row.get(key) for row in self._data])

        def __setitem__(self, key, values):
            for row, val in zip(self._data, values):
                row[key] = val

    pandas_mod = types.SimpleNamespace(DataFrame=_DummyDataFrame)
    import sys
    sys.modules["pandas"] = pandas_mod


_install_pandas_stubs()


def _install_pil_stubs():
    if importlib.util.find_spec("PIL") is not None:
        return

    class _DummyImage:
        def save(self, path, *args, **kwargs):
            from pathlib import Path
            Path(path).write_bytes(b"")

        def convert(self, mode):
            return self

    class _ImageModule:
        @staticmethod
        def open(path):
            return _DummyImage()

        @staticmethod
        def new(mode, size, color=None):
            return _DummyImage()

    pil_mod = types.SimpleNamespace(Image=_ImageModule)
    import sys
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = _ImageModule


_install_pil_stubs()


def _install_torchvision_stubs():
    if importlib.util.find_spec("torchvision") is not None:
        return

    class _IdentityTransform:
        def __call__(self, x):
            return x

    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class _DummyDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.zeros(3, 32, 32), 0

    transforms_mod = types.SimpleNamespace(
        Compose=_Compose,
        RandomCrop=lambda *args, **kwargs: _IdentityTransform(),
        RandomHorizontalFlip=lambda *args, **kwargs: _IdentityTransform(),
        ToTensor=lambda *args, **kwargs: _IdentityTransform(),
        Normalize=lambda *args, **kwargs: _IdentityTransform(),
        Resize=lambda *args, **kwargs: _IdentityTransform(),
        CenterCrop=lambda *args, **kwargs: _IdentityTransform(),
        RandomResizedCrop=lambda *args, **kwargs: _IdentityTransform(),
        ColorJitter=lambda *args, **kwargs: _IdentityTransform(),
    )

    datasets_mod = types.SimpleNamespace(
        CIFAR10=_DummyDataset,
        CIFAR100=_DummyDataset,
        ImageFolder=_DummyDataset,
    )

    torchvision_mod = types.SimpleNamespace(transforms=transforms_mod, datasets=datasets_mod)
    import sys
    sys.modules["torchvision"] = torchvision_mod
    sys.modules["torchvision.transforms"] = transforms_mod
    sys.modules["torchvision.datasets"] = datasets_mod


_install_torchvision_stubs()


@pytest.fixture(autouse=True)
def seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture
def force_cpu(monkeypatch):
    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)


@pytest.fixture
def temp_config_dirs(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "checkpoints"
    tb_dir = tmp_path / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir), raising=False)
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(ckpt_dir), raising=False)
    monkeypatch.setattr(config, "TENSORBOARD_DIR", str(tb_dir), raising=False)
    return tmp_path


def make_block_params(
    out_channels=64,
    groups=4,
    pool_type=0,
    pool_stride=1,
    has_senet=0,
    activation_type=0,
    dropout_rate=0.0,
    skip_type=0,
    kernel_size=3,
    expansion=1,
):
    return BlockParams(
        out_channels=out_channels,
        groups=groups,
        pool_type=pool_type,
        pool_stride=pool_stride,
        has_senet=has_senet,
        activation_type=activation_type,
        dropout_rate=dropout_rate,
        skip_type=skip_type,
        kernel_size=kernel_size,
        expansion=expansion,
    )


def make_encoding(unit_num=2, block_nums=None, concat_last=False, expansion=1):
    if block_nums is None:
        block_nums = [3 for _ in range(unit_num)]
    block_params_list = []
    for bn in block_nums:
        unit_blocks = []
        for idx in range(bn):
            skip_type = 1 if concat_last and idx == bn - 1 else 0
            unit_blocks.append(make_block_params(skip_type=skip_type, expansion=expansion))
        block_params_list.append(unit_blocks)
    return Encoder.encode(unit_num, block_nums, block_params_list)


@pytest.fixture
def simple_encoding():
    return make_encoding(unit_num=2, block_nums=[3, 3], concat_last=False, expansion=1)


@pytest.fixture
def dummy_loader():
    data = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 2, (4,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)
