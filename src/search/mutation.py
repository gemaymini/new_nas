# -*- coding: utf-8 -*-
"""
变异算子模块
实现各种变异操作
"""
import random
import copy
from typing import List, Tuple, Dict
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
        
    def swap_blocks(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        all_positions = []
        for unit_idx, block_num in enumerate(block_nums):
            for block_idx in range(block_num):
                all_positions.append((unit_idx, block_idx))
        if len(all_positions) < 2:
            return encoding, {"op": "swap_blocks", "applied": False, "reason": "not_enough_blocks"}
        pos1, pos2 = random.sample(all_positions, 2)
        block1 = block_params_list[pos1[0]][pos1[1]]
        block2 = block_params_list[pos2[0]][pos2[1]]
        block_params_list[pos1[0]][pos1[1]] = block2
        block_params_list[pos2[0]][pos2[1]] = block1
        detail = {
            "op": "swap_blocks",
            "applied": True,
            "positions": [
                {"unit": pos1[0], "block": pos1[1]},
                {"unit": pos2[0], "block": pos2[1]}
            ]
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def swap_units(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num < 2:
            return encoding, {"op": "swap_units", "applied": False, "reason": "not_enough_units"}
        idx1, idx2 = random.sample(range(unit_num), 2)
        block_nums[idx1], block_nums[idx2] = block_nums[idx2], block_nums[idx1]
        block_params_list[idx1], block_params_list[idx2] = block_params_list[idx2], block_params_list[idx1]
        detail = {
            "op": "swap_units",
            "applied": True,
            "unit_idx1": idx1,
            "unit_idx2": idx2
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def add_unit(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num >= config.MAX_UNIT_NUM:
            return encoding, {"op": "add_unit", "applied": False, "reason": "max_unit_limit"}
        new_block_num = search_space.sample_block_num()
        new_blocks = [search_space.sample_block_params() for _ in range(new_block_num)]
        insert_pos = random.randint(0, unit_num)
        block_nums.insert(insert_pos, new_block_num)
        block_params_list.insert(insert_pos, new_blocks)
        detail = {
            "op": "add_unit",
            "applied": True,
            "insert_pos": insert_pos,
            "new_block_num": new_block_num
        }
        return Encoder.encode(unit_num + 1, block_nums, block_params_list), detail

    def add_block(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        valid_units = [i for i in range(unit_num) if block_nums[i] < config.MAX_BLOCK_NUM]
        if not valid_units:
            return encoding, {"op": "add_block", "applied": False, "reason": "max_block_limit"}
        unit_idx = random.choice(valid_units)
        new_block = search_space.sample_block_params()
        insert_pos = random.randint(0, block_nums[unit_idx])
        block_params_list[unit_idx].insert(insert_pos, new_block)
        block_nums[unit_idx] += 1
        detail = {
            "op": "add_block",
            "applied": True,
            "unit_idx": unit_idx,
            "insert_pos": insert_pos,
            "block_params": new_block.to_list()
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def delete_unit(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num <= config.MIN_UNIT_NUM:
            return encoding, {"op": "delete_unit", "applied": False, "reason": "min_unit_limit"}
        delete_idx = random.randint(0, unit_num - 1)
        removed_blocks = block_params_list[delete_idx]
        del block_nums[delete_idx]
        del block_params_list[delete_idx]
        detail = {
            "op": "delete_unit",
            "applied": True,
            "delete_idx": delete_idx,
            "removed_block_num": len(removed_blocks)
        }
        return Encoder.encode(unit_num - 1, block_nums, block_params_list), detail

    def delete_block(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        valid_units = [i for i in range(unit_num) if block_nums[i] > config.MIN_BLOCK_NUM]
        if not valid_units:
            return encoding, {"op": "delete_block", "applied": False, "reason": "min_block_limit"}
        unit_idx = random.choice(valid_units)
        delete_idx = random.randint(0, block_nums[unit_idx] - 1)
        removed_block = block_params_list[unit_idx][delete_idx]
        del block_params_list[unit_idx][delete_idx]
        block_nums[unit_idx] -= 1
        detail = {
            "op": "delete_block",
            "applied": True,
            "unit_idx": unit_idx,
            "delete_idx": delete_idx,
            "removed_block": removed_block.to_list()
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def modify_block(self, encoding: List[int], num_params_to_modify: int = 3) -> Tuple[List[int], Dict]:
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
        changes = []

        for param in params_to_modify:
            if param == 'out_channels':
                new_value = search_space.sample_channel()
                changes.append({"param": param, "old": old_block.out_channels, "new": new_value})
                old_block.out_channels = new_value
            elif param == 'groups':
                old_val = old_block.groups
                old_block.groups = search_space.sample_groups()
                changes.append({"param": param, "old": old_val, "new": old_block.groups})
            elif param == 'pool_type':
                old_val = old_block.pool_type
                old_block.pool_type = search_space.sample_pool_type()
                changes.append({"param": param, "old": old_val, "new": old_block.pool_type})
            elif param == 'pool_stride':
                old_val = old_block.pool_stride
                old_block.pool_stride = search_space.sample_pool_stride()
                changes.append({"param": param, "old": old_val, "new": old_block.pool_stride})
            elif param == 'has_senet':
                old_val = old_block.has_senet
                old_block.has_senet = search_space.sample_senet()
                changes.append({"param": param, "old": old_val, "new": old_block.has_senet})
            elif param == 'activation_type':
                old_val = old_block.activation_type
                old_block.activation_type = search_space.sample_activation()
                changes.append({"param": param, "old": old_val, "new": old_block.activation_type})
            elif param == 'dropout_rate':
                old_val = old_block.dropout_rate
                old_block.dropout_rate = search_space.sample_dropout()
                changes.append({"param": param, "old": old_val, "new": old_block.dropout_rate})
            elif param == 'skip_type':
                old_val = old_block.skip_type
                old_block.skip_type = search_space.sample_skip_type()
                changes.append({"param": param, "old": old_val, "new": old_block.skip_type})
            elif param == 'kernel_size':
                old_val = old_block.kernel_size
                old_block.kernel_size = search_space.sample_kernel_size()
                changes.append({"param": param, "old": old_val, "new": old_block.kernel_size})
            
        detail = {
            "op": "modify_block",
            "applied": True,
            "unit_idx": unit_idx,
            "block_idx": block_idx,
            "changes": changes
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def mutate(self, individual: Individual) -> Individual:
        new_encoding = copy.deepcopy(individual.encoding)
        applied_ops = []
        mutation_applied = False
        
        # 使用固定概率进行变异操作
        if random.random() < self.prob_swap_blocks:
            new_encoding, detail = self.swap_blocks(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_swap_units:
            new_encoding, detail = self.swap_units(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_add_unit:
            new_encoding, detail = self.add_unit(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_add_block:
            new_encoding, detail = self.add_block(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_delete_unit:
            new_encoding, detail = self.delete_unit(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_delete_block:
            new_encoding, detail = self.delete_block(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
        if random.random() < self.prob_modify_block:
            new_encoding, detail = self.modify_block(new_encoding)
            applied_ops.append(detail)
            mutation_applied = mutation_applied or detail.get("applied", False)
            
        if not mutation_applied:
            new_encoding, detail = self.modify_block(new_encoding)
            applied_ops.append(detail)
            
        new_individual = Individual(new_encoding)
        new_individual.op_history = [{
            "type": "mutation",
            "parent_id": individual.id,
            "ops": applied_ops
        }]
        logger.log_mutation("mutate", individual.id, new_individual.id, applied_ops)
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
    def uniform_unit_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual, Dict]:
        unit_num1, block_nums1, block_params_list1 = Encoder.decode(parent1.encoding)
        unit_num2, block_nums2, block_params_list2 = Encoder.decode(parent2.encoding)
        
        new_unit_num = random.randint(
            max(config.MIN_UNIT_NUM, min(unit_num1, unit_num2)),
            min(config.MAX_UNIT_NUM, max(unit_num1, unit_num2))
        )
        
        c1_nums, c1_params = [], []
        c2_nums, c2_params = [], []
        unit_sources = []
        
        for i in range(new_unit_num):
            select_from_p1 = random.random() < 0.5
            unit_sources.append("p1" if select_from_p1 else "p2")
            
            # Helper to get block info from parent or random
            def get_unit_info(p_nums, p_params, idx):
                if idx < len(p_nums):
                    return p_nums[idx], copy.deepcopy(p_params[idx])
                else:
                    nb = search_space.sample_block_num()
                    return nb, [search_space.sample_block_params() for _ in range(nb)]

            bn1, bp1 = get_unit_info(block_nums1, block_params_list1, i)
            bn2, bp2 = get_unit_info(block_nums2, block_params_list2, i)
            
            if select_from_p1:
                c1_nums.append(bn1); c1_params.append(bp1)
                c2_nums.append(bn2); c2_params.append(bp2)
            else:
                c1_nums.append(bn2); c1_params.append(bp2)
                c2_nums.append(bn1); c2_params.append(bp1)

        child1 = Individual(Encoder.encode(new_unit_num, c1_nums, c1_params))
        child2 = Individual(Encoder.encode(new_unit_num, c2_nums, c2_params))
        detail = {
            "op": "uniform_unit_crossover",
            "applied": True,
            "unit_sources": unit_sources,
            "new_unit_num": new_unit_num
        }
        return child1, child2, detail

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual, Dict]:
        return self.uniform_unit_crossover(parent1, parent2)

mutation_operator = MutationOperator()
selection_operator = SelectionOperator()
crossover_operator = CrossoverOperator()
