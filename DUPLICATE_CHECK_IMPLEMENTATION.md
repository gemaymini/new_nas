# 个体判重功能实现说明

## 概述
在神经架构搜索过程中实现了个体判重机制，确保种群多样性，避免重复评估相同架构。

## 实现的功能

### 1. 核心机制 (encoding.py)

#### Individual 类增强
- **`__eq__` 方法**: 基于编码比较两个个体是否相同
- **`__hash__` 方法**: 计算个体哈希值，支持使用集合进行高效判重
- **`get_encoding_tuple` 方法**: 获取编码的元组表示，用于判重

```python
def __eq__(self, other: 'Individual') -> bool:
    """判断两个个体是否相等（基于编码）"""
    if not isinstance(other, Individual):
        return False
    return (self.normal_cell.to_list() == other.normal_cell.to_list() and 
            self.reduction_cell.to_list() == other.reduction_cell.to_list())

def __hash__(self) -> int:
    """计算个体哈希值（用于去重）"""
    return hash((tuple(self.normal_cell.to_list()), 
                tuple(self.reduction_cell.to_list())))
```

#### DuplicateChecker 工具类
提供三个静态方法用于判重：

1. **`is_duplicate(individual, population)`**: 检查个体是否在种群中重复
2. **`get_encoding_set(population)`**: 获取种群的编码集合（用于批量判重）
3. **`is_duplicate_fast(individual, encoding_set)`**: 使用预计算的集合快速判重

### 2. 配置参数 (config.py)

```python
# 判重配置参数
ENABLE_DUPLICATE_CHECK = True          # 是否启用判重
MAX_DUPLICATE_REPAIR_ATTEMPTS = 10     # 修复重复个体的最大尝试次数
MAX_INIT_DUPLICATE_ATTEMPTS = 100      # 初始化时避免重复的最大尝试次数
```

### 3. 种群初始化判重 (search_space.py)

#### PopulationInitializer 新增方法
- **`create_unique_individual(existing_population, max_attempts)`**: 创建不与现有种群重复的个体

工作流程：
1. 预先计算现有种群的编码集合
2. 循环生成新个体并检查是否重复
3. 如果在 `MAX_INIT_DUPLICATE_ATTEMPTS` 次尝试内未找到唯一个体，允许使用重复个体
4. 确保始终返回有效个体（不会返回 None）

### 4. 进化过程判重 (evolution.py)

#### 初始化阶段
在 `initialize_population()` 方法中：
- 使用 `create_unique_individual()` 确保初始种群无重复
- 维护 `existing_population` 列表用于判重

```python
def initialize_population(self):
    logger.info("Initializing population...")
    existing_population = list(self.population)
    
    while len(self.population) < self.population_size:
        ind = population_initializer.create_unique_individual(existing_population)
        # ... 评估、添加到种群
        existing_population.append(ind)
```

#### 进化阶段
在 `_generate_offspring()` 方法中实现判重：

1. **提前检查**: 如果禁用判重，直接生成并返回
2. **编码集合优化**: 只计算一次当前种群的编码集合
3. **循环尝试**: 最多尝试 `MAX_DUPLICATE_REPAIR_ATTEMPTS` 次
4. **失败容错**: 超过最大次数后允许使用重复个体

新增辅助方法 `_create_child()` 用于执行交叉和变异操作，提高代码可读性。

```python
def _generate_offspring(self, parent1, parent2):
    if not config.ENABLE_DUPLICATE_CHECK:
        return self._create_child(parent1, parent2)
    
    encoding_set = DuplicateChecker.get_encoding_set(list(self.population))
    
    for attempt in range(max_attempts):
        child = self._create_child(parent1, parent2)
        if not child.validate():
            child = self._repair_individual(child, [parent1, parent2])
        
        if not DuplicateChecker.is_duplicate_fast(child, encoding_set):
            return child
    
    # 允许重复
    return child
```

## 性能优化

### 1. 哈希优化
- 使用 Python 内置的 `__hash__` 和 `__eq__` 方法
- 集合查找时间复杂度：O(1)

### 2. 批量判重优化
- 预先计算编码集合，避免重复转换
- 在循环中只计算一次，而不是每次都计算

### 3. 早期退出
- 提前检查 `ENABLE_DUPLICATE_CHECK` 标志
- 找到唯一个体后立即返回

## 日志记录

### 信息日志
- 初始化阶段创建唯一个体时的尝试次数
- 进化阶段生成唯一后代时的尝试次数

### 调试日志
- 检测到重复个体时的详细信息

### 警告日志
- 超过最大尝试次数，允许使用重复个体

## 使用方式

### 启用判重（默认）
```python
# config.py
ENABLE_DUPLICATE_CHECK = True
MAX_DUPLICATE_REPAIR_ATTEMPTS = 10
MAX_INIT_DUPLICATE_ATTEMPTS = 100
```

### 禁用判重
```python
# config.py
ENABLE_DUPLICATE_CHECK = False
```

### 调整尝试次数
```python
# 增加容忍度，允许更多尝试
MAX_DUPLICATE_REPAIR_ATTEMPTS = 20
MAX_INIT_DUPLICATE_ATTEMPTS = 200
```

## 测试

运行测试脚本验证判重功能：
```bash
cd src
uv run python test_duplicate_check.py
```

测试内容：
1. 个体相等性比较
2. 哈希计算
3. 基本判重功能
4. 快速判重功能
5. 禁用判重功能

## 注意事项

1. **性能影响**: 判重会增加一定的计算开销，但通过优化（哈希、集合）已最小化
2. **搜索空间大小**: 在搜索空间较小时，可能经常遇到重复，建议适当增加 `MAX_DUPLICATE_REPAIR_ATTEMPTS`
3. **容错机制**: 算法设计为即使无法找到唯一个体也能继续运行，确保搜索不会中断
4. **日志级别**: 设置 `LOG_LEVEL = 'DEBUG'` 可以看到更详细的判重信息

## 代码质量改进

### 解决的问题
1. ✅ 避免 `create_unique_individual` 返回 None
2. ✅ 提取 `_create_child` 方法避免代码重复
3. ✅ 简化 `DuplicateChecker.is_duplicate` 使用 `__eq__` 方法
4. ✅ 优化 `_generate_offspring` 只计算一次编码集合
5. ✅ 添加详细的文档字符串

### 代码结构
- 职责清晰：判重逻辑集中在 `DuplicateChecker`
- 解耦合：通过配置开关可以完全禁用判重
- 可测试：每个方法职责单一，易于测试

## 后续优化建议

1. **历史记录判重**: 考虑在 `run_screening_and_training()` 中已经实现了基于 genotype 的去重
2. **统计信息**: 可以添加重复率统计，帮助评估搜索空间探索情况
3. **自适应尝试次数**: 根据搜索空间大小和种群规模动态调整尝试次数
