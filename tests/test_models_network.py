# -*- coding: utf-8 -*-
"""
Tests for models.network.
"""
import torch

from core.encoding import BlockParams
from models.network import get_activation, SEBlock, RegBlock, RegUnit, SearchedNetwork, NetworkBuilder

from conftest import make_block_params, make_encoding


def test_get_activation_types():
    assert isinstance(get_activation(0), torch.nn.ReLU)
    assert isinstance(get_activation(1), torch.nn.SiLU)
    assert isinstance(get_activation(99), torch.nn.ReLU)


def test_seblock_shape():
    block = SEBlock(channels=8, reduction=2)
    x = torch.randn(2, 8, 4, 4)
    y = block(x)
    assert y.shape == x.shape


def test_regblock_skip_add_shape():
    bp = make_block_params(skip_type=0, expansion=1)
    block = RegBlock(in_channels=16, block_params=bp)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape[1] == block.final_out_channels


def test_regblock_skip_concat_shape():
    bp = make_block_params(skip_type=1, expansion=1)
    block = RegBlock(in_channels=16, block_params=bp)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape[1] == block.final_out_channels


def test_regblock_skip_none_shape():
    bp = make_block_params(skip_type=2, expansion=2)
    block = RegBlock(in_channels=16, block_params=bp)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape[1] == 128


def test_regunit_output_channels():
    params = [make_block_params(expansion=1), make_block_params(expansion=1)]
    unit = RegUnit(in_channels=16, block_params_list=params)
    assert unit.get_output_channels() > 0


def test_searched_network_forward():
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    net = SearchedNetwork(encoding, input_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, 10)
    assert net.get_param_count() > 0


def test_network_builder_param_count():
    encoding = make_encoding(unit_num=2, block_nums=[3, 3])
    count = NetworkBuilder.calculate_param_count(encoding, input_channels=3, num_classes=10)
    assert count > 0
