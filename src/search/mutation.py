# -*- coding: utf-8 -*-
"""
变异算子模块
实现各种变异操作
"""
import random
import copy
from typing import List, Tuple
from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import search_space
from utils.logger import logger

class MutationOperator:
    def __init__(self):
        # 使用全局固定的变异概率
        self.prob_swap_blocks = config.PROB_SWAP_BLOCKS
        self.prob_swap_units = config.PROB_SWAP_UNITS
        self.prob_add_unit = config.PROB_ADD_UNIT
        self.prob_add_block = config.PROB_ADD_BLOCK
        self.prob_delete_unit = config.PROB_DELETE_UNIT
        self.prob_delete_block = config.PROB_DELETE_BLOCK
        self.prob_modify_block = config.PROB_MODIFY_BLOCK
        
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

    def modify_block(self, encoding: List[int], num_params_to_modify: int = 3) -> List[int]:
        """
        修改一个 block 的多个参数
        
        Args:
            encoding: 编码列表
            num_params_to_modify: 要修改的参数数量，默认为3
        """
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        unit_idx = random.randint(0, unit_num - 1)
        block_idx = random.randint(0, block_nums[unit_idx] - 1)
        old_block = block_params_list[unit_idx][block_idx]
        
        # 所有可修改的参数列表
        all_params = [
            'out_channels', 'groups', 'pool_type', 'pool_stride', 'has_senet',
            'activation_type', 'dropout_rate', 'skip_type', 'kernel_size'
        ]
        
        # 随机选择 num_params_to_modify 个参数进行变异
        params_to_modify = random.sample(all_params, num_params_to_modify)

        for param in params_to_modify:
            if param == 'out_channels':
                new_value = search_space.sample_channel()
                old_block.out_channels = new_value
            elif param == 'groups':
                old_block.groups = search_space.sample_groups()
            elif param == 'pool_type':
                old_block.pool_type = search_space.sample_pool_type()
            elif param == 'pool_stride':
                old_block.pool_stride = search_space.sample_pool_stride()
            elif param == 'has_senet':
                old_block.has_senet = search_space.sample_senet()
            elif param == 'activation_type':
                old_block.activation_type = search_space.sample_activation()
            elif param == 'dropout_rate':
                old_block.dropout_rate = search_space.sample_dropout()
            elif param == 'skip_type':
                old_block.skip_type = search_space.sample_skip_type()
            elif param == 'kernel_size':
                old_block.kernel_size = search_space.sample_kernel_size()
            
        return Encoder.encode(unit_num, block_nums, block_params_list)

    def mutate(self, individual: Individual) -> Individual:
        new_encoding = copy.deepcopy(individual.encoding)
        mutation_applied = False
        applied_mutations = []
        
        # 记录原始状态
        original_unit_num, original_block_nums, _ = Encoder.decode(individual.encoding)
        
        # 使用固定概率进行变异操作
        if random.random() < self.prob_swap_blocks:
            new_encoding = self.swap_blocks(new_encoding)
            mutation_applied = True
            applied_mutations.append('swap_blocks')
            
        if random.random() < self.prob_swap_units:
            new_encoding = self.swap_units(new_encoding)
            mutation_applied = True
            applied_mutations.append('swap_units')
            
        if random.random() < self.prob_add_unit:
            new_encoding = self.add_unit(new_encoding)
            mutation_applied = True
            applied_mutations.append('add_unit')
            
        if random.random() < self.prob_add_block:
            new_encoding = self.add_block(new_encoding)
            mutation_applied = True
            applied_mutations.append('add_block')
            
        if random.random() < self.prob_delete_unit:
            new_encoding = self.delete_unit(new_encoding)
            mutation_applied = True
            applied_mutations.append('delete_unit')
            
        if random.random() < self.prob_delete_block:
            new_encoding = self.delete_block(new_encoding)
            mutation_applied = True
            applied_mutations.append('delete_block')
            
        if random.random() < self.prob_modify_block:
            new_encoding = self.modify_block(new_encoding)
            mutation_applied = True
            applied_mutations.append('modify_block')
            
        if not mutation_applied:
            new_encoding = self.modify_block(new_encoding)
            applied_mutations.append('modify_block')
            
        new_individual = Individual(new_encoding)
        
        # 记录变异后状态
        new_unit_num, new_block_nums, _ = Encoder.decode(new_encoding)
        
        # 记录详细的变异信息
        mutation_details = {
            'applied_mutations': applied_mutations,
            'original_structure': {
                'unit_num': original_unit_num,
                'block_nums': original_block_nums,
                'total_blocks': sum(original_block_nums)
            },
            'new_structure': {
                'unit_num': new_unit_num,
                'block_nums': new_block_nums,
                'total_blocks': sum(new_block_nums)
            },
            'encoding_length_change': len(new_encoding) - len(individual.encoding)
        }
        
        logger.log_detailed_mutation(
            operation_type='combined_mutation',
            parent_id=str(individual.id),
            child_id=str(new_individual.id),
            details=mutation_details
        )
        
        return new_individual

class SelectionOperator:
    def tournament_selection(self, population: List[Individual], tournament_size: int = None, num_winners: int = None) -> List[Individual]:
        if tournament_size is None: tournament_size = config.TOURNAMENT_SIZE
        if num_winners is None: num_winners = config.TOURNAMENT_WINNERS
        
        tournament_size = min(tournament_size, len(population))
        competitors = random.sample(population, tournament_size)
        # fitness 越小越好，所以升序排列，取前面的作为胜者
        sorted_competitors = sorted(competitors, key=lambda x: x.fitness if x.fitness is not None else float('inf'), reverse=False)
        return sorted_competitors[:num_winners]

class CrossoverOperator:
    def uniform_unit_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        unit_num1, block_nums1, block_params_list1 = Encoder.decode(parent1.encoding)
        unit_num2, block_nums2, block_params_list2 = Encoder.decode(parent2.encoding)
        
        new_unit_num = random.randint(
            max(config.MIN_UNIT_NUM, min(unit_num1, unit_num2)),
            min(config.MAX_UNIT_NUM, max(unit_num1, unit_num2))
        )
        
        c1_nums, c1_params = [], []
        c2_nums, c2_params = [], []
        
        # 记录交叉详细信息
        unit_selections = []  # 记录每个unit位置选择了哪个父个体
        generated_units = []  # 记录哪些位置生成了新的unit
        
        for i in range(new_unit_num):
            select_from_p1 = random.random() < 0.5
            
            # Helper to get block info from parent or random
            def get_unit_info(p_nums, p_params, idx):
                if idx < len(p_nums):
                    return p_nums[idx], copy.deepcopy(p_params[idx])
                else:
                    generated_units.append(idx)
                    nb = search_space.sample_block_num()
                    return nb, [search_space.sample_block_params() for _ in range(nb)]

            bn1, bp1 = get_unit_info(block_nums1, block_params_list1, i)
            bn2, bp2 = get_unit_info(block_nums2, block_params_list2, i)
            
            if select_from_p1:
                c1_nums.append(bn1); c1_params.append(bp1)
                c2_nums.append(bn2); c2_params.append(bp2)
                unit_selections.append({'position': i, 'child1_from': 'parent1', 'child2_from': 'parent2'})
            else:
                c1_nums.append(bn2); c1_params.append(bp2)
                c2_nums.append(bn1); c2_params.append(bp1)
                unit_selections.append({'position': i, 'child1_from': 'parent2', 'child2_from': 'parent1'})

        child1 = Individual(Encoder.encode(new_unit_num, c1_nums, c1_params))
        child2 = Individual(Encoder.encode(new_unit_num, c2_nums, c2_params))
        
        # 记录交叉详细信息
        crossover_details = {
            'parent_structures': {
                'parent1': {'unit_num': unit_num1, 'block_nums': block_nums1, 'total_blocks': sum(block_nums1)},
                'parent2': {'unit_num': unit_num2, 'block_nums': block_nums2, 'total_blocks': sum(block_nums2)}
            },
            'child_structures': {
                'child1': {'unit_num': new_unit_num, 'block_nums': c1_nums, 'total_blocks': sum(c1_nums)},
                'child2': {'unit_num': new_unit_num, 'block_nums': c2_nums, 'total_blocks': sum(c2_nums)}
            },
            'crossover_info': {
                'new_unit_num': new_unit_num,
                'unit_selections': unit_selections,
                'generated_units': generated_units,
                'selection_probability': 0.5
            }
        }
        
        logger.log_detailed_crossover(
            parent1_id=str(parent1.id),
            parent2_id=str(parent2.id),
            child1_id=str(child1.id),
            child2_id=str(child2.id),
            details=crossover_details
        )
        
        return child1, child2

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        return self.uniform_unit_crossover(parent1, parent2)

mutation_operator = MutationOperator()
selection_operator = SelectionOperator()
crossover_operator = CrossoverOperator()
