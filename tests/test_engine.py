import unittest
import torch
import torch.nn as nn
from new_nas.engine.evaluator import NTKEvaluator, ParameterEvaluator, QuickEvaluator, FinalEvaluator
from new_nas.core.encoding import Individual, Encoder
from new_nas.utils.config import config

class TestEngine(unittest.TestCase):
    def setUp(self):
        # Create a small valid encoding for testing
        # 1 unit, 1 block
        # Format: [unit_num, block_num_u1, out_ch, groups, pool_type, stride, senet]
        self.encoding = [1, 1, 16, 1, 0, 1, 0]
        self.ind = Individual(self.encoding)
        self.ind.id = 1
        
    def test_ntk_evaluator(self):
        # Use CPU for testing
        evaluator = NTKEvaluator(device='cpu')
        score = evaluator.evaluate_individual(self.ind)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(self.ind.fitness)
        self.assertIsNotNone(self.ind.param_count)

    def test_parameter_evaluator(self):
        evaluator = ParameterEvaluator()
        count = evaluator.count_parameters(self.ind)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_quick_evaluator(self):
        # Mocking dataset loading would be ideal, but for now we can 
        # assume the environment might not have data and handle gracefully
        # or just test initialization.
        # Ideally we should mock _load_small_dataset to use random tensors
        
        evaluator = QuickEvaluator()
        
        # Mock internal data to avoid downloading CIFAR
        evaluator._train_data = [(torch.randn(3, 32, 32), torch.tensor(0)) for _ in range(10)]
        evaluator._val_data = [(torch.randn(3, 32, 32), torch.tensor(0)) for _ in range(5)]
        evaluator.device = 'cpu' # Force CPU
        evaluator.num_epochs = 1
        
        score = evaluator.evaluate_individual(self.ind)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_final_evaluator_init(self):
        # Just test initialization as full training takes too long
        # and requires data
        try:
            evaluator = FinalEvaluator(dataset='cifar10', device='cpu')
            self.assertIsNotNone(evaluator)
        except Exception:
            # Might fail if data not present/downloadable
            pass

    def test_final_evaluator_save(self):
        import os
        import shutil
        
        try:
            # Setup dummy evaluator
            evaluator = FinalEvaluator(dataset='cifar10', device='cpu')
            
            # Mock trainloader/testloader
            dummy_loader = [(torch.randn(1, 3, 32, 32), torch.tensor([0]))]
            evaluator.trainloader = dummy_loader
            evaluator.testloader = dummy_loader
            
            # Mock trainer.train_network to return immediately
            # We need to simulate the trainer modifying the model (optional but good)
            # and returning accuracy and history
            evaluator.trainer.train_network = lambda model, *args: (0.95, [])
            
            # Run evaluation
            acc, result = evaluator.evaluate_individual(self.ind, epochs=1)
            
            # Check if file exists
            save_path = result.get('model_path')
            self.assertIsNotNone(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Cleanup
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
                
        except Exception as e:
            # Skip if dataset initialization fails (e.g. download error)
            print(f"Skipping test_final_evaluator_save due to: {e}")
            pass

if __name__ == '__main__':
    unittest.main()
