# -*- coding: utf-8 -*-
"""
Tests for search.mutation.
"""
import random

from configuration.config import config
from core.encoding import Encoder, Individual
from search.mutation import MutationOperator
from utils.logger import logger

from conftest import make_encoding, make_block_params


def _assert_concat_last(encoding):
    _, _, block_params_list = Encoder.decode(encoding)
    for unit_blocks in block_params_list:
        for idx, bp in enumerate(unit_blocks):
            if bp.skip_type == 1:
                assert idx == len(unit_blocks) - 1


def test_swap_blocks():
    op = MutationOperator()
    encoding = make_encoding(unit_num=2, block_nums=[3, 3], concat_last=True)
    new_encoding, detail = op.swap_blocks(encoding)
    assert detail["op"] == "swap_blocks"
    assert isinstance(new_encoding, list)
    _assert_concat_last(new_encoding)


def test_swap_units():
    op = MutationOperator()
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    new_encoding, detail = op.swap_units(encoding)
    assert detail["op"] == "swap_units"
    assert isinstance(new_encoding, list)


def test_add_unit_rejects_max(monkeypatch):
    op = MutationOperator()
    encoding = make_encoding(unit_num=config.MAX_UNIT_NUM, block_nums=[3, 3, 3, 3])
    new_encoding, detail = op.add_unit(encoding)
    assert detail["applied"] is False
    assert new_encoding == encoding


def test_add_block_rejects_max():
    op = MutationOperator()
    encoding = make_encoding(unit_num=2, block_nums=[config.MAX_BLOCK_NUM, config.MAX_BLOCK_NUM])
    new_encoding, detail = op.add_block(encoding)
    assert detail["applied"] is False
    assert new_encoding == encoding


def test_delete_unit_rejects_min():
    op = MutationOperator()
    encoding = make_encoding(unit_num=config.MIN_UNIT_NUM, block_nums=[3, 3])
    new_encoding, detail = op.delete_unit(encoding)
    assert detail["applied"] is False
    assert new_encoding == encoding


def test_delete_block_rejects_min():
    op = MutationOperator()
    encoding = make_encoding(unit_num=2, block_nums=[config.MIN_BLOCK_NUM, config.MIN_BLOCK_NUM])
    new_encoding, detail = op.delete_block(encoding)
    assert detail["applied"] is False
    assert new_encoding == encoding


def test_modify_block_changes():
    random.seed(1)
    op = MutationOperator()
    encoding = make_encoding(unit_num=2, block_nums=[3, 3], concat_last=True)
    new_encoding, detail = op.modify_block(encoding, num_params_to_modify=3)
    assert detail["applied"] is True
    assert len(detail["changes"]) == 3
    _assert_concat_last(new_encoding)


def test_mutate_fallback(monkeypatch):
    op = MutationOperator()
    op.prob_swap_blocks = 0.0
    op.prob_swap_units = 0.0
    op.prob_add_unit = 0.0
    op.prob_add_block = 0.0
    op.prob_delete_unit = 0.0
    op.prob_delete_block = 0.0
    op.prob_modify_block = 0.0
    monkeypatch.setattr(logger, "log_mutation", lambda *args, **kwargs: None)
    ind = Individual(make_encoding(unit_num=2, block_nums=[3, 3]))
    mutated = op.mutate(ind)
    assert mutated.op_history
    assert mutated.op_history[0]["type"] == "mutation"
