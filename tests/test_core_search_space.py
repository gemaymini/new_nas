# -*- coding: utf-8 -*-
"""
Tests for core.search_space.
"""
import random

from configuration.config import config
from core.search_space import SearchSpace, PopulationInitializer
from core.encoding import Encoder, Individual

from conftest import make_encoding


def test_search_space_sampling_ranges():
    ss = SearchSpace()
    assert ss.sample_channel() in config.CHANNEL_OPTIONS
    assert ss.sample_groups() in config.GROUP_OPTIONS
    assert ss.sample_pool_type() in config.POOL_TYPE_OPTIONS
    assert ss.sample_pool_stride() in config.POOL_STRIDE_OPTIONS
    assert ss.sample_senet() in config.SENET_OPTIONS
    assert ss.sample_activation() in config.ACTIVATION_OPTIONS
    assert ss.sample_dropout() in config.DROPOUT_OPTIONS
    assert ss.sample_kernel_size() in config.KERNEL_SIZE_OPTIONS
    assert ss.sample_expansion() in config.EXPANSION_OPTIONS


def test_sample_skip_type_no_concat():
    ss = SearchSpace()
    for _ in range(10):
        assert ss.sample_skip_type(allow_concat=False) != 1


def test_sample_block_params_no_concat():
    ss = SearchSpace()
    bp = ss.sample_block_params(allow_concat=False)
    assert bp.skip_type != 1


def test_population_initializer_valid_individual(monkeypatch):
    random.seed(0)
    ss = SearchSpace()
    pi = PopulationInitializer(ss)

    encoding = make_encoding(unit_num=2, block_nums=[3, 3])

    def fake_create():
        return encoding

    monkeypatch.setattr(pi, "_create_constrained_encoding", fake_create)
    ind = pi.create_valid_individual()
    assert isinstance(ind, Individual)
    assert Encoder.validate_encoding(ind.encoding)


def test_constrained_encoding_concat_last(monkeypatch):
    random.seed(1)
    ss = SearchSpace()
    pi = PopulationInitializer(ss)
    encoding = pi._create_constrained_encoding()
    unit_num, block_nums, block_params_list = Encoder.decode(encoding)
    for unit_blocks in block_params_list:
        for idx, bp in enumerate(unit_blocks):
            if bp.skip_type == 1:
                assert idx == len(unit_blocks) - 1

