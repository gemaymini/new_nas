# -*- coding: utf-8 -*-
"""
变异算子模块
实现各种变异操作，支持自适应变异
"""
import random
import copy
from typing import List, Tuple
from new_nas.utils.config import config
from new_nas.core.encoding import Encoder, Individual
from new_nas.core.search_space import search_space
from new_nas.utils.logger import logger

class AdaptiveMutationController:
    def __init__(self):
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.current_scale = 1.0
        
    def update(self, current_gen: int, best_fitness: float, phase: int):
        if phase == 1:
            phase_scale = config.MUTATION_SCALE_PHASE1
        elif phase == 2:
            phase_scale = config.MUTATION_SCALE_PHASE2
        else:
            phase_scale = config.MUTATION_SCALE_PHASE3
            
        if self.best_fitness_history:
            if best_fitness <= max(self.best_fitness_history[-config.STAGNATION_THRESHOLD:]):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
        self.best_fitness_history.append(best_fitness)
        
        if self.stagnation_counter >= config.STAGNATION_THRESHOLD:
            self.current_scale = phase_scale * config.STAGNATION_MUTATION_BOOST
        else:
            self.current_scale = phase_scale
        return self.current_scale

    def get_scaled_probability(self, base_prob: float) -> float:
        if not config.ADAPTIVE_MUTATION: return base_prob
        return min(1.0, base_prob * self.current_scale)
        
    def reset(self):
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.current_scale = 1.0

adaptive_mutation_controller = AdaptiveMutationController()

class MutationOperator:
    def __init__(self):
        self.base_prob_swap_blocks = config.PROB_SWAP_BLOCKS
        self.base_prob_swap_units = config.PROB_SWAP_UNITS
        self.base_prob_add_unit = config.PROB_ADD_UNIT
        self.base_prob_add_block = config.PROB_ADD_BLOCK
        self.base_prob_delete_unit = config.PROB_DELETE_UNIT
        self.base_prob_delete_block = config.PROB_DELETE_BLOCK
        self.base_prob_modify_block = config.PROB_MODIFY_BLOCK
        
    @property
    def prob_swap_blocks(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_swap_blocks)
    @property
    def prob_swap_units(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_swap_units)
    @property
    def prob_add_unit(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_add_unit)
    @property
    def prob_add_block(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_add_block)
    @property
    def prob_delete_unit(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_delete_unit)
    @property
    def prob_delete_block(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_delete_block)
    @property
    def prob_modify_block(self): return adaptive_mutation_controller.get_scaled_probability(self.base_prob_modify_block)

    def swap_blocks(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        all_positions = []
        for unit_idx, block_num in enumerate(block_nums):
            for block_idx in range(block_num):
                all_positions.append((unit_idx, block_idx))
        if len(all_positions) < 2: return encoding
        pos1, pos2 = random.sample(all_positions, 2)
        block1 = block_params_list[pos1[0]][pos1[1]]
        block2 = block_params_list[pos2[0]][pos2[1]]
        block_params_list[pos1[0]][pos1[1]] = block2
        block_params_list[pos2[0]][pos2[1]] = block1
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def swap_units(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num < 2: return encoding
        idx1, idx2 = random.sample(range(unit_num), 2)
        block_nums[idx1], block_nums[idx2] = block_nums[idx2], block_nums[idx1]
        block_params_list[idx1], block_params_list[idx2] = block_params_list[idx2], block_params_list[idx1]
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def add_unit(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num >= config.MAX_UNIT_NUM: return encoding
        new_block_num = search_space.sample_block_num()
        new_blocks = [search_space.sample_block_params() for _ in range(new_block_num)]
        insert_pos = random.randint(0, unit_num)
        block_nums.insert(insert_pos, new_block_num)
        block_params_list.insert(insert_pos, new_blocks)
        return Encoder.encode(unit_num + 1, block_nums, block_params_list)

    def add_block(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        valid_units = [i for i in range(unit_num) if block_nums[i] < config.MAX_BLOCK_NUM]
        if not valid_units: return encoding
        unit_idx = random.choice(valid_units)
        new_block = search_space.sample_block_params()
        insert_pos = random.randint(0, block_nums[unit_idx])
        block_params_list[unit_idx].insert(insert_pos, new_block)
        block_nums[unit_idx] += 1
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def delete_unit(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num <= config.MIN_UNIT_NUM: return encoding
        delete_idx = random.randint(0, unit_num - 1)
        del block_nums[delete_idx]
        del block_params_list[delete_idx]
        return Encoder.encode(unit_num - 1, block_nums, block_params_list)

    def delete_block(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        valid_units = [i for i in range(unit_num) if block_nums[i] > config.MIN_BLOCK_NUM]
        if not valid_units: return encoding
        unit_idx = random.choice(valid_units)
        delete_idx = random.randint(0, block_nums[unit_idx] - 1)
        del block_params_list[unit_idx][delete_idx]
        block_nums[unit_idx] -= 1
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def modify_block(self, encoding: List[int]) -> List[int]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        unit_idx = random.randint(0, unit_num - 1)
        block_idx = random.randint(0, block_nums[unit_idx] - 1)
        old_block = block_params_list[unit_idx][block_idx]
        param_to_modify = random.choice(['out_channels', 'groups', 'pool_type', 'pool_stride', 'has_senet'])
        
        if param_to_modify == 'out_channels':
            new_value = search_space.sample_channel()
            old_block.out_channels = new_value
            if old_block.groups > new_value: old_block.groups = search_space.sample_groups(new_value)
        elif param_to_modify == 'groups':
            old_block.groups = search_space.sample_groups(old_block.out_channels)
        elif param_to_modify == 'pool_type':
            old_block.pool_type = search_space.sample_pool_type()
        elif param_to_modify == 'pool_stride':
            old_block.pool_stride = search_space.sample_pool_stride()
        elif param_to_modify == 'has_senet':
            old_block.has_senet = search_space.sample_senet()
            
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def mutate(self, individual: Individual, current_gen: int) -> Individual:
        new_encoding = copy.deepcopy(individual.encoding)
        mutation_applied = False
        
        if random.random() < self.prob_swap_blocks:
            new_encoding = self.swap_blocks(new_encoding); mutation_applied = True
        if random.random() < self.prob_swap_units:
            new_encoding = self.swap_units(new_encoding); mutation_applied = True
        if random.random() < self.prob_add_unit:
            new_encoding = self.add_unit(new_encoding); mutation_applied = True
        if random.random() < self.prob_add_block:
            new_encoding = self.add_block(new_encoding); mutation_applied = True
        if random.random() < self.prob_delete_unit:
            new_encoding = self.delete_unit(new_encoding); mutation_applied = True
        if random.random() < self.prob_delete_block:
            new_encoding = self.delete_block(new_encoding); mutation_applied = True
        if random.random() < self.prob_modify_block:
            new_encoding = self.modify_block(new_encoding); mutation_applied = True
            
        if not mutation_applied:
            new_encoding = self.modify_block(new_encoding)
            
        new_individual = Individual(new_encoding)
        new_individual.birth_generation = current_gen
        logger.log_mutation("mutate", individual.id, new_individual.id)
        return new_individual

class SelectionOperator:
    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int = None, num_winners: int = None) -> List[Individual]:
        if tournament_size is None: tournament_size = config.TOURNAMENT_SIZE
        if num_winners is None: num_winners = config.TOURNAMENT_WINNERS
        
        tournament_size = min(tournament_size, len(population))
        competitors = random.sample(population, tournament_size)
        sorted_competitors = sorted(competitors, key=lambda x: x.fitness if x.fitness else float('-inf'), reverse=True)
        return sorted_competitors[:num_winners]

class CrossoverOperator:
    @staticmethod
    def uniform_unit_crossover(parent1: Individual, parent2: Individual, current_gen: int) -> Tuple[Individual, Individual]:
        unit_num1, block_nums1, block_params_list1 = Encoder.decode(parent1.encoding)
        unit_num2, block_nums2, block_params_list2 = Encoder.decode(parent2.encoding)
        
        new_unit_num = random.randint(
            max(config.MIN_UNIT_NUM, min(unit_num1, unit_num2)),
            min(config.MAX_UNIT_NUM, max(unit_num1, unit_num2))
        )
        
        c1_nums, c1_params = [], []
        c2_nums, c2_params = [], []
        
        for i in range(new_unit_num):
            select_from_p1 = random.random() < 0.5
            
            # Helper to get block info from parent or random
            def get_block_info(p_nums, p_params, idx):
                if idx < len(p_nums):
                    return p_nums[idx], copy.deepcopy(p_params[idx])
                else:
                    nb = search_space.sample_block_num()
                    return nb, [search_space.sample_block_params() for _ in range(nb)]

            bn1, bp1 = get_block_info(block_nums1, block_params_list1, i)
            bn2, bp2 = get_block_info(block_nums2, block_params_list2, i)
            
            if select_from_p1:
                c1_nums.append(bn1); c1_params.append(bp1)
                c2_nums.append(bn2); c2_params.append(bp2)
            else:
                c1_nums.append(bn2); c1_params.append(bp2)
                c2_nums.append(bn1); c2_params.append(bp1)

        child1 = Individual(Encoder.encode(new_unit_num, c1_nums, c1_params))
        child1.birth_generation = current_gen
        child2 = Individual(Encoder.encode(new_unit_num, c2_nums, c2_params))
        child2.birth_generation = current_gen
        return child1, child2

    def crossover(self, parent1: Individual, parent2: Individual, current_gen: int) -> Tuple[Individual, Individual]:
        # Simplify to just uniform for now as it's most robust
        return self.uniform_unit_crossover(parent1, parent2, current_gen)

mutation_operator = MutationOperator()
selection_operator = SelectionOperator()
crossover_operator = CrossoverOperator()
