# -*- coding: utf-8 -*-
"""
Tests for core.encoding.
"""
import pytest

from configuration.config import config
from core.encoding import BlockParams, Encoder

from conftest import make_block_params, make_encoding


def test_block_params_roundtrip():
    bp = BlockParams(
        out_channels=64,
        groups=4,
        pool_type=0,
        pool_stride=1,
        has_senet=1,
        activation_type=0,
        dropout_rate=0.1,
        skip_type=2,
        kernel_size=5,
        expansion=2,
    )
    encoded = bp.to_list()
    decoded = BlockParams.from_list(encoded)
    assert decoded.out_channels == 64
    assert decoded.groups == 4
    assert decoded.pool_type == 0
    assert decoded.pool_stride == 1
    assert decoded.has_senet == 1
    assert decoded.activation_type == 0
    assert decoded.dropout_rate == 0.1
    assert decoded.skip_type == 2
    assert decoded.kernel_size == 5
    assert decoded.expansion == 2


def test_encode_decode_roundtrip(simple_encoding):
    unit_num, block_nums, block_params_list = Encoder.decode(simple_encoding)
    reencoded = Encoder.encode(unit_num, block_nums, block_params_list)
    assert reencoded == simple_encoding


def test_validate_encoding_rejects_bad_unit_count(simple_encoding):
    bad = simple_encoding[:]
    bad[0] = config.MIN_UNIT_NUM - 1
    assert Encoder.validate_encoding(bad) is False


def test_validate_encoding_rejects_bad_block_count():
    encoding = make_encoding(unit_num=2, block_nums=[2, 2])
    assert Encoder.validate_encoding(encoding) is False


def test_validate_encoding_rejects_concat_not_last():
    block_nums = [3, 3]
    block_params_list = []
    for bn in block_nums:
        unit_blocks = []
        for idx in range(bn):
            skip_type = 1 if idx == 0 else 0
            unit_blocks.append(make_block_params(skip_type=skip_type))
        block_params_list.append(unit_blocks)
    encoding = Encoder.encode(len(block_nums), block_nums, block_params_list)
    assert Encoder.validate_encoding(encoding) is False


def test_validate_channel_count_rejects_large(monkeypatch, simple_encoding):
    monkeypatch.setattr(config, "MAX_CHANNELS", 32, raising=False)
    assert Encoder.validate_channel_count(simple_encoding) is False


def test_validate_feature_size_rejects_small(monkeypatch):
    monkeypatch.setattr(config, "MIN_FEATURE_SIZE", 2, raising=False)
    block_nums = [3]
    block_params_list = [[
        make_block_params(pool_stride=2),
        make_block_params(pool_stride=2),
        make_block_params(pool_stride=2),
    ]]
    encoding = Encoder.encode(1, block_nums, block_params_list)
    assert Encoder.validate_feature_size(encoding, input_size=4) is False


def test_print_architecture_output(simple_encoding, capsys):
    Encoder.print_architecture(simple_encoding)
    out = capsys.readouterr().out
    assert "INFO: architecture" in out
    assert "INFO: unit" in out
    assert "INFO: block" in out

