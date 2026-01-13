# -*- coding: utf-8 -*-
"""
Tests for main entrypoint.
"""
import types

import torch

import main


def test_set_seed():
    main.set_seed(123)
    assert torch.initial_seed() is not None


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    args = main.parse_args()
    assert args.dataset in ("cifar10", "cifar100", "imagenet")


def test_main_happy_path(monkeypatch):
    dummy_args = types.SimpleNamespace(
        dataset="cifar10",
        imagenet_root="/tmp",
        seed=1,
        resume=None,
        no_final_eval=False,
    )
    monkeypatch.setattr(main, "parse_args", lambda: dummy_args)
    monkeypatch.setattr(main, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.logger, "setup_file_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.tb_logger, "setup", lambda *args, **kwargs: None)

    class DummyNAS:
        def __init__(self):
            self.loaded = False

        def load_checkpoint(self, path):
            self.loaded = True

        def run_search(self):
            return None

        def run_screening_and_training(self):
            return None

    monkeypatch.setattr(main, "AgingEvolutionNAS", DummyNAS)
    main.main()
