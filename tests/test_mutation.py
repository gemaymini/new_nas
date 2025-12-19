import unittest
from new_nas.search.mutation import MutationOperator, CrossoverOperator, AdaptiveMutationController
from new_nas.core.encoding import Encoder, Individual
from new_nas.utils.config import config

class TestMutation(unittest.TestCase):
    def setUp(self):
        self.mut_op = MutationOperator()
        self.cross_op = CrossoverOperator()
        # Create a standard individual for testing
        # 2 units, 2 blocks each
        self.base_encoding = [2, 2, 2, 
                              16, 1, 0, 1, 0, 16, 1, 0, 1, 0, 
                              32, 1, 1, 1, 0, 32, 1, 1, 1, 0]
        self.ind = Individual(self.base_encoding)

    def test_swap_blocks(self):
        new_enc = self.mut_op.swap_blocks(list(self.base_encoding))
        self.assertTrue(Encoder.validate_encoding(new_enc))
        # Length should be same
        self.assertEqual(len(new_enc), len(self.base_encoding))

    def test_add_unit(self):
        # Only add if not max
        if self.base_encoding[0] < config.MAX_UNIT_NUM:
            new_enc = self.mut_op.add_unit(list(self.base_encoding))
            self.assertTrue(Encoder.validate_encoding(new_enc))
            self.assertEqual(new_enc[0], self.base_encoding[0] + 1)

    def test_delete_unit(self):
        # Only delete if > min
        if self.base_encoding[0] > config.MIN_UNIT_NUM:
            new_enc = self.mut_op.delete_unit(list(self.base_encoding))
            self.assertTrue(Encoder.validate_encoding(new_enc))
            self.assertEqual(new_enc[0], self.base_encoding[0] - 1)

    def test_modify_block(self):
        # This one is random, but should always produce valid encoding
        new_enc = self.mut_op.modify_block(list(self.base_encoding))
        self.assertTrue(Encoder.validate_encoding(new_enc))

    def test_crossover(self):
        parent1 = Individual(self.base_encoding)
        # Create different parent
        enc2 = list(self.base_encoding)
        enc2[0] = 2
        enc2[1] = 1 # Unit 1 has 1 block
        # Remove parameters for one block (5 ints)
        enc2 = enc2[:3] + enc2[8:] 
        
        # Verify parent2 is valid first
        if not Encoder.validate_encoding(enc2):
            # Fallback to safe encoding if manual manipulation failed
            enc2 = Encoder.create_random_encoding()
            
        parent2 = Individual(enc2)
        
        child1, child2 = self.cross_op.crossover(parent1, parent2, current_gen=1)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
        self.assertTrue(Encoder.validate_encoding(child1.encoding))
        self.assertTrue(Encoder.validate_encoding(child2.encoding))

    def test_adaptive_controller(self):
        controller = AdaptiveMutationController()
        scale = controller.update(current_gen=1, best_fitness=0.5, phase=1)
        self.assertEqual(scale, config.MUTATION_SCALE_PHASE1)
        
        # Simulate stagnation
        for _ in range(config.STAGNATION_THRESHOLD + 1):
            scale = controller.update(current_gen=1, best_fitness=0.5, phase=1)
            
        self.assertGreater(scale, config.MUTATION_SCALE_PHASE1)

if __name__ == '__main__':
    unittest.main()
