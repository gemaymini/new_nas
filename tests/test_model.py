import unittest
import torch
import torch.nn as nn
from new_nas.model.network import SEBlock, ConvUnit, RegBlock, RegUnit, SearchedNetwork
from new_nas.core.encoding import BlockParams

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_size = (1, 3, 32, 32)
        self.device = 'cpu'

    def test_se_block(self):
        channels = 32
        se = SEBlock(channels, reduction=4)
        x = torch.randn(1, channels, 16, 16)
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_conv_unit(self):
        conv = ConvUnit(in_channels=3, out_channels=16)
        x = torch.randn(self.input_size)
        out = conv(x)
        self.assertEqual(out.shape, (1, 16, 32, 32))

    def test_reg_block_stride1(self):
        # Stride 1, no channel change
        params = BlockParams(out_channels=32, groups=1, pool_type=0, pool_stride=1, has_senet=0)
        block = RegBlock(in_channels=32, block_params=params)
        x = torch.randn(1, 32, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 32, 32, 32))

    def test_reg_block_stride2(self):
        # Stride 2, channels double (in_channels * stride)
        params = BlockParams(out_channels=32, groups=1, pool_type=1, pool_stride=2, has_senet=1)
        block = RegBlock(in_channels=16, block_params=params)
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        # Output channels = in_channels * stride = 16 * 2 = 32
        self.assertEqual(out.shape, (1, 32, 16, 16))

    def test_reg_unit(self):
        # Unit with 2 blocks
        params1 = BlockParams(out_channels=16, groups=1, pool_type=0, pool_stride=1, has_senet=0)
        params2 = BlockParams(out_channels=32, groups=2, pool_type=1, pool_stride=2, has_senet=1)
        unit = RegUnit(in_channels=16, block_params_list=[params1, params2])
        
        x = torch.randn(1, 16, 32, 32)
        out = unit(x)
        # Block 1: 16 -> 16 (stride 1)
        # Block 2: 16 -> 32 (stride 2)
        self.assertEqual(out.shape, (1, 32, 16, 16))

    def test_searched_network(self):
        # 2 Units, 1 block each
        # Unit 1: stride 1
        # Unit 2: stride 2
        encoding = [2, 1, 1,
                    16, 1, 0, 1, 0,
                    32, 1, 1, 2, 0]
        
        net = SearchedNetwork(encoding, num_classes=10)
        x = torch.randn(self.input_size)
        out = net(x)
        self.assertEqual(out.shape, (1, 10))
        
        param_count = net.get_param_count()
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)

if __name__ == '__main__':
    unittest.main()
