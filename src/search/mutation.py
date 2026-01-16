# -*- coding: utf-8 -*-
"""
Mutation, selection, and crossover operators.
"""
import random
import copy
from typing import List, Tuple, Dict
from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import search_space
from utils.logger import logger


class MutationOperator:
    """
    Handles mutation operations on architecture encodings.
    """
    def __init__(self):
        self.prob_swap_blocks = config.PROB_SWAP_BLOCKS
        self.prob_swap_units = config.PROB_SWAP_UNITS
        self.prob_add_unit = config.PROB_ADD_UNIT
        self.prob_add_block = config.PROB_ADD_BLOCK
        self.prob_delete_unit = config.PROB_DELETE_UNIT
        self.prob_delete_block = config.PROB_DELETE_BLOCK
        self.prob_modify_block = config.PROB_MODIFY_BLOCK

    def _repair_channels(self, block_params_list: List[List]) -> None:
        """
        Scan and repair out_channels to strictly follow 64/1024 positional rules.
        """
        unit_num = len(block_params_list)
        for unit_idx, unit_blocks in enumerate(block_params_list):
            for bp in unit_blocks:
                # Rule: 64 only in Unit 0
                if unit_idx > 0 and bp.out_channels == 64:
                    bp.out_channels = search_space.sample_channel(unit_idx, unit_num)
                
                # Rule: 1024 only in Last Unit
                if unit_idx < unit_num - 1 and bp.out_channels == 1024:
                    bp.out_channels = search_space.sample_channel(unit_idx, unit_num)

    def _enforce_concat_last(self, block_params_list: List[List]) -> None:
        """
        Ensure 'concat' skip type is ONLY present in the last block of a unit.
        Also triggers channel repair.
        
        Args:
            block_params_list: Nested list of block parameters.
        """
        # First repair channels since unit counts/positions might have changed
        self._repair_channels(block_params_list)

        # Ensure concat skips appear only in the last block of each unit.
        for unit_blocks in block_params_list:
            last_idx = len(unit_blocks) - 1
            for idx, bp in enumerate(unit_blocks):
                if idx != last_idx and bp.skip_type == config.SKIP_TYPE_CONCAT:
                    bp.skip_type = search_space.sample_skip_type(allow_concat=False)

    def swap_blocks(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly swap two blocks in the architecture."""
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
        self._enforce_concat_last(block_params_list)
        detail = {
            "op": "swap_blocks",
            "applied": True,
            "positions": [
                {"unit": pos1[0], "block": pos1[1]},
                {"unit": pos2[0], "block": pos2[1]},
            ],
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def swap_units(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly swap two full units in the architecture."""
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num < 2:
            return encoding, {"op": "swap_units", "applied": False, "reason": "not_enough_units"}
        idx1, idx2 = random.sample(range(unit_num), 2)
        block_nums[idx1], block_nums[idx2] = block_nums[idx2], block_nums[idx1]
        block_params_list[idx1], block_params_list[idx2] = block_params_list[idx2], block_params_list[idx1]
        self._enforce_concat_last(block_params_list)
        detail = {
            "op": "swap_units",
            "applied": True,
            "unit_idx1": idx1,
            "unit_idx2": idx2,
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def add_unit(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly add a new unit to the architecture."""
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num >= config.MAX_UNIT_NUM:
            return encoding, {"op": "add_unit", "applied": False, "reason": "max_unit_limit"}
        new_block_num = search_space.sample_block_num()
        insert_pos = random.randint(0, unit_num)
        new_blocks = [
            search_space.sample_block_params(
                allow_concat=idx == new_block_num - 1,
                unit_idx=insert_pos,
                total_units=unit_num + 1
            )
            for idx in range(new_block_num)
        ]
        block_nums.insert(insert_pos, new_block_num)
        block_params_list.insert(insert_pos, new_blocks)
        self._enforce_concat_last(block_params_list)
        detail = {
            "op": "add_unit",
            "applied": True,
            "insert_pos": insert_pos,
            "new_block_num": new_block_num,
        }
        return Encoder.encode(unit_num + 1, block_nums, block_params_list), detail

    def add_block(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly add a new block to an existing unit."""
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        valid_units = [i for i in range(unit_num) if block_nums[i] < config.MAX_BLOCK_NUM]
        if not valid_units:
            return encoding, {"op": "add_block", "applied": False, "reason": "max_block_limit"}
        unit_idx = random.choice(valid_units)
        insert_pos = random.randint(0, block_nums[unit_idx])
        allow_concat = insert_pos == block_nums[unit_idx]
        new_block = search_space.sample_block_params(
            allow_concat=allow_concat,
            unit_idx=unit_idx,
            total_units=unit_num
        )
        block_params_list[unit_idx].insert(insert_pos, new_block)
        block_nums[unit_idx] += 1
        self._enforce_concat_last(block_params_list)
        detail = {
            "op": "add_block",
            "applied": True,
            "unit_idx": unit_idx,
            "insert_pos": insert_pos,
            "block_params": new_block.to_list(),
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def delete_unit(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly delete a unit from the architecture."""
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        if unit_num <= config.MIN_UNIT_NUM:
            return encoding, {"op": "delete_unit", "applied": False, "reason": "min_unit_limit"}
        delete_idx = random.randint(0, unit_num - 1)
        removed_blocks = block_params_list[delete_idx]
        del block_nums[delete_idx]
        del block_params_list[delete_idx]
        self._enforce_concat_last(block_params_list)
        detail = {
            "op": "delete_unit",
            "applied": True,
            "delete_idx": delete_idx,
            "removed_block_num": len(removed_blocks),
        }
        return Encoder.encode(unit_num - 1, block_nums, block_params_list), detail

    def delete_block(self, encoding: List[int]) -> Tuple[List[int], Dict]:
        """Randomly delete a block from a unit."""
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
            "removed_block": removed_block.to_list(),
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def modify_block(self, encoding: List[int], num_params_to_modify: int = 3) -> Tuple[List[int], Dict]:
        """Modify multiple parameters of one block."""
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        unit_idx = random.randint(0, unit_num - 1)
        block_idx = random.randint(0, block_nums[unit_idx] - 1)
        old_block = block_params_list[unit_idx][block_idx]
        # Only the last block in a unit may use concat skips.
        allow_concat = block_idx == block_nums[unit_idx] - 1

        # Define mapping of param name -> (getter, sampler)
        # getter: lambda block: block.field
        # sampler: lambda: search_space.sample_field(...)
        # We process 'skip_type' specially due to allow_concat.

        param_map = {
            # "out_channels" handled separately
            "groups": (lambda b: b.groups, search_space.sample_groups),
            "pool_type": (lambda b: b.pool_type, search_space.sample_pool_type),
            "pool_stride": (lambda b: b.pool_stride, search_space.sample_pool_stride),
            "has_senet": (lambda b: b.has_senet, search_space.sample_senet),
            "activation_type": (lambda b: b.activation_type, search_space.sample_activation),
            "dropout_rate": (lambda b: b.dropout_rate, search_space.sample_dropout),
            "kernel_size": (lambda b: b.kernel_size, search_space.sample_kernel_size),
            "expansion": (lambda b: b.expansion, search_space.sample_expansion),
        }

        # Select params to modify excluding skip_type initially
        available_keys = list(param_map.keys()) + ["skip_type", "out_channels"]
        params_to_modify = random.sample(available_keys, num_params_to_modify)
        changes = []

        for param in params_to_modify:
            if param == "skip_type":
                old_val = old_block.skip_type
                new_val = search_space.sample_skip_type(allow_concat=allow_concat)
                old_block.skip_type = new_val
                changes.append({"param": param, "old": old_val, "new": new_val})
            elif param == "out_channels":
                old_val = old_block.out_channels
                new_val = search_space.sample_channel(unit_idx=unit_idx, total_units=unit_num)
                old_block.out_channels = new_val
                changes.append({"param": param, "old": old_val, "new": new_val})
            else:
                getter, sampler = param_map[param]
                old_val = getter(old_block)
                new_val = sampler()
                
                # Update attribute directly
                setattr(old_block, param, new_val)
                changes.append({"param": param, "old": old_val, "new": new_val})

        self._enforce_concat_last(block_params_list)

        detail = {
            "op": "modify_block",
            "applied": True,
            "unit_idx": unit_idx,
            "block_idx": block_idx,
            "changes": changes,
        }
        return Encoder.encode(unit_num, block_nums, block_params_list), detail

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply a random set of mutations to an individual.

        Args:
            individual (Individual): The individual to mutate.
        
        Returns:
            Individual: A new, mutated individual.
        """
        new_encoding = copy.deepcopy(individual.encoding)
        applied_ops = []
        mutation_applied = False

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
            "ops": applied_ops,
        }]
        logger.log_mutation("mutate", individual.id, new_individual.id, applied_ops)
        return new_individual


class SelectionOperator:
    """
    Handles parent selection for evolution.
    """
    def tournament_selection(
        self,
        population: List[Individual],
        tournament_size: int = None,
        num_winners: int = None,
    ) -> List[Individual]:
        """
        Select best individuals from random tournaments.

        Args:
            population (List[Individual]): Candidate pool.
            tournament_size (int, optional): Size of tournament.
            num_winners (int, optional): Number of winners to select.

        Returns:
            List[Individual]: Selected parents.
        """
        if tournament_size is None:
            tournament_size = config.TOURNAMENT_SIZE
        if num_winners is None:
            num_winners = config.TOURNAMENT_WINNERS

        tournament_size = min(tournament_size, len(population))
        competitors = random.sample(population, tournament_size)
        sorted_competitors = sorted(
            competitors,
            key=lambda x: x.fitness if x.fitness is not None else float("inf"),
            reverse=False,
        )
        return sorted_competitors[:num_winners]


class CrossoverOperator:
    """
    Handles crossover operations to combine two parents.
    """
    def uniform_unit_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual, Dict]:
        """
        Perform uniform crossover at the unit level.

        Args:
            parent1 (Individual): First parent.
            parent2 (Individual): Second parent.

        Returns:
            Tuple[Individual, Individual, Dict]: Two children and operation details.
        """
        unit_num1, block_nums1, block_params_list1 = Encoder.decode(parent1.encoding)
        unit_num2, block_nums2, block_params_list2 = Encoder.decode(parent2.encoding)

        new_unit_num = random.randint(
            max(config.MIN_UNIT_NUM, min(unit_num1, unit_num2)),
            min(config.MAX_UNIT_NUM, max(unit_num1, unit_num2)),
        )

        c1_nums, c1_params = [], []
        c2_nums, c2_params = [], []
        unit_sources = []

        for i in range(new_unit_num):
            select_from_p1 = random.random() < 0.5
            unit_sources.append("p1" if select_from_p1 else "p2")

            def get_unit_info(p_nums, p_params, idx):
                if idx < len(p_nums):
                    return p_nums[idx], copy.deepcopy(p_params[idx])
                nb = search_space.sample_block_num()
                return nb, [
                    search_space.sample_block_params(
                        allow_concat=j == nb - 1,
                        unit_idx=idx,
                        total_units=new_unit_num
                    )
                    for j in range(nb)
                ]

            bn1, bp1 = get_unit_info(block_nums1, block_params_list1, i)
            bn2, bp2 = get_unit_info(block_nums2, block_params_list2, i)

            if select_from_p1:
                c1_nums.append(bn1)
                c1_params.append(bp1)
                c2_nums.append(bn2)
                c2_params.append(bp2)
            else:
                c1_nums.append(bn2)
                c1_params.append(bp2)
                c2_nums.append(bn1)
                c2_params.append(bp1)

        # Repair channels and skips for children
        MutationOperator()._enforce_concat_last(c1_params)
        MutationOperator()._enforce_concat_last(c2_params)

        child1 = Individual(Encoder.encode(new_unit_num, c1_nums, c1_params))
        child2 = Individual(Encoder.encode(new_unit_num, c2_nums, c2_params))
        detail = {
            "op": "uniform_unit_crossover",
            "applied": True,
            "unit_sources": unit_sources,
            "new_unit_num": new_unit_num,
        }
        return child1, child2, detail

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual, Dict]:
        """
        Main entry point for crossover.
        """
        return self.uniform_unit_crossover(parent1, parent2)


mutation_operator = MutationOperator()
selection_operator = SelectionOperator()
crossover_operator = CrossoverOperator()
