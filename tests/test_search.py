import unittest
from new_nas.search.nsga2 import NSGAII
from new_nas.core.encoding import Individual
from new_nas.search.evolution import EvolutionaryNAS
from new_nas.utils.config import config

class TestSearch(unittest.TestCase):
    def test_nsga2_dominance(self):
        # Ind 1: better in all
        ind1 = Individual([])
        ind1.survival_time = 1  # -1
        ind1.param_count = 100  # -log(100)
        ind1.quick_score = 0.9  # 0.9
        
        # Ind 2: worse in all
        ind2 = Individual([])
        ind2.survival_time = 10 # -10 (worse)
        ind2.param_count = 1000 # -log(1000) (worse)
        ind2.quick_score = 0.5  # 0.5 (worse)
        
        self.assertTrue(NSGAII._dominates(ind1, ind2))
        self.assertFalse(NSGAII._dominates(ind2, ind1))

    def test_nsga2_non_dominated_sort(self):
        # Create a population where dominance is clear
        ind1 = Individual([]); ind1.id=1
        ind1.survival_time=1; ind1.param_count=100; ind1.quick_score=0.9
        
        ind2 = Individual([]); ind2.id=2
        ind2.survival_time=10; ind2.param_count=1000; ind2.quick_score=0.5
        
        ind3 = Individual([]); ind3.id=3 # Non-dominated with ind1 maybe?
        ind3.survival_time=1; ind3.param_count=1000; ind3.quick_score=0.95 # Better score, worse params
        
        pop = [ind1, ind2, ind3]
        fronts = NSGAII.fast_non_dominated_sort(pop)
        
        # ind1 should be in front 0
        # ind2 should be dominated by ind1
        self.assertIn(ind1, fronts[0])
        # ind3 might be front 0 too because it has better score
        
        # ind2 is definitely worse than ind1
        self.assertNotIn(ind2, fronts[0])

    def test_evolution_init(self):
        # Test small evolution run
        # Use small population to be fast
        nas = EvolutionaryNAS(population_size=4, max_gen=1, g1=0, g2=1)
        nas.initialize_population()
        self.assertEqual(len(nas.population), 4)
        self.assertIsNotNone(nas.best_individual)
        
    def test_evolution_step(self):
        nas = EvolutionaryNAS(population_size=4, max_gen=2, g1=0, g2=1)
        nas.initialize_population()
        start_gen = nas.current_gen
        nas.evolve_one_generation()
        self.assertEqual(nas.current_gen, start_gen + 1)
        # Population size should remain constant
        self.assertEqual(len(nas.population), 4)

if __name__ == '__main__':
    unittest.main()
