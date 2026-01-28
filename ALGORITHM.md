# DARTS-based Neural Architecture Search

基于 DARTS 搜索空间的遗传算法神经网络架构搜索系统。

## 算法概述

本系统使用 **遗传算法 (Genetic Algorithm)** 在 **DARTS 搜索空间** 中搜索最优的网络架构。

### 核心概念

#### 1. Cell 结构
网络由两种 Cell 构成：
- **Normal Cell**: 保持特征分辨率
- **Reduction Cell**: 降低分辨率，通道数翻倍

每个 Cell 是一个 DAG (有向无环图):
```
Input_0 (c_{k-2}) ──┐
                    ├──→ Node_0 ──→ Node_1 ──→ ... ──→ Node_N ──→ Concat ──→ Output
Input_1 (c_{k-1}) ──┘
```

#### 2. 操作集合
每条边代表一个操作，共 8 种：

| ID | 操作 | 说明 |
|----|------|------|
| 0 | `zero` | 无连接 |
| 1 | `skip_connect` | 恒等映射 |
| 2 | `sep_conv_3x3` | 3×3 深度可分离卷积 |
| 3 | `sep_conv_5x5` | 5×5 深度可分离卷积 |
| 4 | `dil_conv_3x3` | 3×3 空洞卷积 (dilation=2) |
| 5 | `dil_conv_5x5` | 5×5 空洞卷积 (dilation=2) |
| 6 | `max_pool_3x3` | 3×3 最大池化 |
| 7 | `avg_pool_3x3` | 3×3 平均池化 |

#### 3. 编码方式
每个 Cell 使用固定长度整数列表编码：
```
[source_0, op_0, source_1, op_1, ...]
```
- `source`: 输入来源索引 (0=Input_0, 1=Input_1, 2+=中间节点)
- `op`: 操作 ID (0-7)

**编码长度**: `NUM_NODES × EDGES_PER_NODE × 2 = 4 × 2 × 2 = 16`

### 网络结构

```
┌─────────────────────────────────────────┐
│                  Stem                    │  3 → 108 channels
├─────────────────────────────────────────┤
│  Normal Cell × 4                        │  Stage 1
├─────────────────────────────────────────┤
│  Reduction Cell                         │  分辨率 ↓，通道 ↑
├─────────────────────────────────────────┤
│  Normal Cell × 4                        │  Stage 2
├─────────────────────────────────────────┤
│  Reduction Cell                         │  分辨率 ↓，通道 ↑
├─────────────────────────────────────────┤
│  Normal Cell × 4                        │  Stage 3
├─────────────────────────────────────────┤
│  Global Avg Pool → FC                   │  分类器
└─────────────────────────────────────────┘
```

## 遗传算法

### 流程

```
1. 初始化种群 (100 个随机个体)
2. NTK 评估每个个体的 fitness
3. 进化循环:
   a. 锦标赛选择 2 个父代
   b. 交叉产生子代
   c. 变异子代
   d. 评估子代 fitness
   e. 加入种群，移除最老个体
4. 筛选 Top-N 进行短期训练
5. 筛选 Top-K 进行完整训练
```

### 变异算子

#### 1. 边操作变异 (`mutate_edge_operation`)
随机选择一条边，将其操作替换为另一个操作。

```python
# 示例: sep_conv_3x3 → max_pool_3x3
edge.op_id = random.choice([0,1,2,3,4,5,6,7])
```

#### 2. 节点输入变异 (`mutate_edge_source`)
随机选择一条边，将其输入来源替换为另一个有效来源。

```python
# 示例: Node_1 输入从 Input_0 改为 Node_0
edge.source = random.choice(valid_sources)
```

### 交叉算子

Cell 级别交叉：随机从两个父代选择 Normal/Reduction Cell。

```python
child.normal_cell = parent1.normal_cell if random.random() < 0.5 else parent2.normal_cell
child.reduction_cell = parent1.reduction_cell if random.random() < 0.5 else parent2.reduction_cell
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_NODES` | 4 | Cell 中间节点数 |
| `EDGES_PER_NODE` | 2 | 每节点输入边数 |
| `INIT_CHANNELS` | 36 | 初始通道数 |
| `CELLS_PER_STAGE` | 4 | 每 Stage 的 Normal Cell 数 |
| `NUM_STAGES` | 3 | Stage 数量 |
| `POPULATION_SIZE` | 100 | 种群大小 |
| `MAX_GEN` | 5000 | 最大进化代数 |
| `TOURNAMENT_SIZE` | 5 | 锦标赛大小 |

## 使用方法

### 运行搜索
```bash
cd src
uv run python main.py --population_size 100 --max_gen 5000 --dataset cifar10
```

### 参数说明
- `--population_size`: 种群大小
- `--max_gen`: 最大进化代数
- `--dataset`: 数据集 (cifar10/cifar100)
- `--seed`: 随机种子
- `--resume`: 从 checkpoint 恢复

### 测试模块
```bash
cd src
uv run python test_darts.py
```

## 项目结构

```
src/
├── configuration/
│   └── config.py          # 配置参数
├── core/
│   ├── encoding.py        # Cell 编码
│   └── search_space.py    # 搜索空间
├── models/
│   └── network.py         # DARTS 网络
├── search/
│   ├── mutation.py        # 变异/交叉算子
│   └── evolution.py       # 进化算法
├── engine/
│   ├── evaluator.py       # NTK/训练评估
│   └── trainer.py         # 网络训练
└── main.py                # 主入口
```

## 输出

- `logs/ntk_history.json`: NTK 历史记录
- `logs/ntk_curve.png`: NTK 曲线图
- `checkpoints/`: 进化过程 checkpoint
- `checkpoints/final_models/`: 最终训练模型

## 参考文献

- DARTS: Differentiable Architecture Search (Liu et al., 2019)
- Regularized Evolution for Image Classifier Architecture Search (Real et al., 2019)
