# -*- coding: utf-8 -*-
"""
DARTS 重构测试脚本
验证编码、变异、网络构建模块
"""
import sys
import torch

print('=' * 60)
print('DARTS 重构测试')
print('=' * 60)

# Test 1: Config
print('\n[Test 1] 配置加载...')
from configuration.config import config
print(f'  NUM_NODES={config.NUM_NODES}')
print(f'  EDGES_PER_NODE={config.EDGES_PER_NODE}')
print(f'  Operations: {len(config.OPERATIONS)} types')
print(f'  INIT_CHANNELS={config.INIT_CHANNELS}')
print('  ✓ Config loaded')

# Test 2: Cell Encoding
print('\n[Test 2] Cell 编码...')
from core.encoding import CellEncoding, Individual, Encoder

cell = CellEncoding()
encoding = cell.to_list()
print(f'  Cell encoding length: {len(encoding)}')
print(f'  Expected length: {CellEncoding.encoding_length()}')

# Validate cell
assert cell.validate(), 'Cell validation failed!'
print(f'  Cell validation: True')

# Test decode/encode
decoded = CellEncoding.from_list(encoding)
assert decoded.to_list() == encoding, 'Encoding/Decoding mismatch!'
print('  ✓ Cell encoding test passed')

# Test 3: Individual
print('\n[Test 3] Individual...')
ind = Individual()
print(f'  Individual id: {ind.id}')
print(f'  Normal cell valid: {ind.normal_cell.validate()}')
print(f'  Reduction cell valid: {ind.reduction_cell.validate()}')

# Test copy
ind_copy = ind.copy()
assert ind_copy.id != ind.id, 'Copy should have different id'
assert ind_copy.normal_cell.to_list() == ind.normal_cell.to_list(), 'Copy mismatch'
print('  ✓ Individual copy test passed')

# Test to_dict/from_dict
ind_dict = ind.to_dict()
ind_restored = Individual.from_dict(ind_dict)
assert ind_restored.normal_cell.to_list() == ind.normal_cell.to_list(), 'Dict restore mismatch'
print('  ✓ Individual serialization test passed')

# Test genotype
genotype = Encoder.get_genotype(ind)
print(f'  Genotype normal edges: {len(genotype["normal"])}')
print(f'  Genotype reduce edges: {len(genotype["reduce"])}')
print('  ✓ Individual test passed')

# Test 4: Search Space
print('\n[Test 4] 搜索空间...')
from core.search_space import search_space, population_initializer

ind2 = search_space.sample_individual()
assert ind2.validate(), 'Sampled individual not valid'
print('  ✓ Search space sampling test passed')

# Test population initializer
ind3 = population_initializer.create_valid_individual()
assert ind3 is not None and ind3.validate(), 'Population initializer failed'
print('  ✓ Population initializer test passed')

# Test 5: Mutation
print('\n[Test 5] 变异算子...')
from search.mutation import mutation_operator, crossover_operator

# Test mutation
mutated = mutation_operator.mutate(ind)
assert mutated.validate(), 'Mutated individual not valid'
print('  ✓ Mutation test passed')

# Test crossover
child = crossover_operator.crossover(ind, ind2)
assert child.validate(), 'Crossover child not valid'
print('  ✓ Crossover test passed')

# Test 6: Network Building
print('\n[Test 6] 网络构建...')
from models.network import NetworkBuilder, DARTSNetwork

network = NetworkBuilder.build_from_individual(ind, num_classes=10)
param_count = network.get_param_count()
print(f'  Network built successfully')
print(f'  Parameter count: {param_count:,}')

# Test forward
x = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    output = network(x)
print(f'  Input shape: {x.shape}')
print(f'  Output shape: {output.shape}')
assert output.shape == (2, 10), f'Output shape mismatch: {output.shape}'
print('  ✓ Network forward test passed')

# Test NetworkBuilder static method
forward_ok = NetworkBuilder.test_forward(ind)
assert forward_ok, 'NetworkBuilder.test_forward failed'
print('  ✓ NetworkBuilder.test_forward passed')

print('\n' + '=' * 60)
print('所有测试通过!')
print('=' * 60)
