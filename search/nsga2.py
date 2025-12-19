# -*- coding: utf-8 -*-
"""
NSGA-II多目标优化模块
"""
import math
from typing import List
from core.encoding import Individual
from utils.logger import logger

class NSGAII:
    """
    NSGA-II相关操作类
    """
    @staticmethod
    def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
        fronts = [[]]
        for p in population:
            p.dominated_solutions = []
            p.domination_count = 0
            for q in population:
                if NSGAII._dominates(p, q):
                    p.dominated_solutions.append(q)
                elif NSGAII._dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        fronts.pop()
        return fronts

    @staticmethod
    def _dominates(p: Individual, q: Individual) -> bool:
        obj_p = NSGAII._get_objectives(p)
        obj_q = NSGAII._get_objectives(q)
        
        better_in_at_least_one = False
        for i in range(len(obj_p)):
            if obj_p[i] < obj_q[i]: return False
            if obj_p[i] > obj_q[i]: better_in_at_least_one = True
        return better_in_at_least_one

    @staticmethod
    def _get_objectives(individual: Individual) -> List[float]:
        # Objective 1: Survival time (minimize -> maximize negative)
        survival_obj = -individual.survival_time
        
        # Objective 2: Parameter count (minimize -> maximize negative log)
        if individual.param_count is None or individual.param_count <= 0:
            param_obj = 0
        else:
            param_obj = -math.log(individual.param_count + 1)
            
        # Objective 3: Quick score (maximize)
        quick_obj = individual.quick_score if individual.quick_score is not None else 0
        
        return [survival_obj, param_obj, quick_obj]

    @staticmethod
    def get_pareto_front(population: List[Individual]) -> List[Individual]:
        fronts = NSGAII.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []

    @staticmethod
    def calculate_crowding_distance(front: List[Individual]):
        if len(front) == 0: return
        
        for ind in front:
            ind.crowding_distance = 0.0
            
        num_objectives = 3
        
        for m in range(num_objectives):
            front.sort(key=lambda x: NSGAII._get_objectives(x)[m])
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_min = NSGAII._get_objectives(front[0])[m]
            obj_max = NSGAII._get_objectives(front[-1])[m]
            obj_range = obj_max - obj_min
            
            if obj_range == 0: continue
            
            for i in range(1, len(front) - 1):
                obj_prev = NSGAII._get_objectives(front[i - 1])[m]
                obj_next = NSGAII._get_objectives(front[i + 1])[m]
                front[i].crowding_distance += (obj_next - obj_prev) / obj_range

    @staticmethod
    def select_by_nsga2(population: List[Individual], num_select: int) -> List[Individual]:
        fronts = NSGAII.fast_non_dominated_sort(population)
        selected = []
        front_idx = 0
        
        while len(selected) + len(fronts[front_idx]) <= num_select:
            NSGAII.calculate_crowding_distance(fronts[front_idx])
            selected.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts): break
            
        if len(selected) < num_select and front_idx < len(fronts):
            NSGAII.calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            remaining = num_select - len(selected)
            selected.extend(fronts[front_idx][:remaining])
            
        return selected
