import unittest
from new_nas.core.encoding import Encoder, BlockParams
from new_nas.core.search_space import search_space
from new_nas.model.network import NetworkBuilder

class TestCore(unittest.TestCase):
    def test_encoding_decoding(self):
        encoding = Encoder.create_random_encoding()
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        re_encoded = Encoder.encode(unit_num, block_nums, block_params_list)
        self.assertEqual(encoding, re_encoded)

    def test_search_space(self):
        block_params = search_space.sample_block_params()
        self.assertIsInstance(block_params, BlockParams)
        
    def test_network_build(self):
        # Create a simple valid encoding
        # 2 units, 1 block each
        encoding = [2, 1, 1, 
                    16, 1, 0, 1, 0, # unit 1 block 1
                    16, 1, 0, 1, 0] # unit 2 block 1
        
        self.assertTrue(Encoder.validate_encoding(encoding))
        
        try:
            network = NetworkBuilder.build_from_encoding(encoding)
            self.assertIsNotNone(network)
        except Exception as e:
            self.fail(f"Network build failed: {e}")

if __name__ == '__main__':
    unittest.main()
