# 神经网络架构搜索 (Neural Architecture Search - NAS)

## 项目概述

本项目实现了一个基于**老化进化算法 (Aging Evolution)** 的神经网络架构搜索系统。该系统通过进化算法自动搜索最优的卷积神经网络架构，使用 **NTK (Neural Tangent Kernel)** 条件数作为快速评估指标（无需完整训练），最终对候选架构进行完整训练以获得高性能模型。

### 主要特点

- 🧬 **老化进化算法**：使用 FIFO 队列实现种群管理，自动淘汰老化个体
- 🔬 **NTK 零成本代理**：基于 NTK 条件数快速评估网络可训练性，无需实际训练
- 🏗️ **灵活的搜索空间**：支持可变数量的 Unit 和 Block，包含 SENet 注意力机制
- 📊 **多阶段筛选**：先 NTK 筛选 → 短期训练验证 → 完整训练最优模型
- 💾 **断点续训**：支持保存和加载 checkpoint，可中断后继续搜索

---

## 项目结构

```
nas/
├── main.py                    # 主程序入口
├── train_topk.py              # 从 checkpoint 训练 Top-K 模型
├── continue_train.py          # 继续训练已有模型
├── requirements.txt           # 依赖包列表
│
├── src/                       # 源代码目录
│   ├── configuration/         # 配置模块
│   │   └── config.py          # 超参数配置
│   │
│   ├── core/                  # 核心模块
│   │   ├── encoding.py        # 编码器与个体类
│   │   └── search_space.py    # 搜索空间定义
│   │
│   ├── data/                  # 数据模块
│   │   └── dataset.py         # 数据集加载器
│   │
│   ├── engine/                # 引擎模块
│   │   ├── evaluator.py       # NTK 评估器 & 最终评估器
│   │   └── trainer.py         # 网络训练器
│   │
│   ├── models/                # 模型模块
│   │   └── network.py         # 网络构建器
│   │
│   ├── search/                # 搜索模块
│   │   ├── evolution.py       # 老化进化算法
│   │   └── mutation.py        # 变异/交叉/选择算子
│   │
│   └── utils/                 # 工具模块
│       └── logger.py          # 日志记录器
│
├── apply/                     # 应用脚本
│   ├── predict.py             # 模型推理
│   └── inspect_model.py       # 查看模型架构
│
├── checkpoints/               # 保存的 checkpoint
│   └── final_models/          # 最终训练的模型
│
├── data/                      # 数据集目录
│   └── cifar-10-batches-py/   # CIFAR-10 数据
│
├── logs/                      # 日志目录
├── runs/                      # TensorBoard 日志
└── test/                      # 测试脚本
```

---

## 安装与配置

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于 GPU 加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖列表

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
psutil>=5.8.0
tensorboard>=2.6.0
tqdm>=4.60.0
```

---

## 使用方法

### 1. 运行架构搜索

```bash
# 默认参数运行
python main.py

# 自定义参数
python main.py --population_size 100 --max_gen 5000 --seed 42

# 测试模式（快速运行，验证代码）
python main.py --test
```

### 2. 从 Checkpoint 训练 Top-K 模型

```bash
python train_topk.py checkpoints/checkpoint_step1000.pkl --top_k 5 --epochs 300
```

### 3. 继续训练已有模型

```bash
python continue_train.py checkpoints/final_models/model_xxx.pth --epochs 100 --lr 0.01
```

### 4. 模型推理

```bash
python apply/predict.py checkpoints/final_models/model_xxx.pth path/to/image.jpg
```

### 5. 查看模型架构

```bash
python apply/inspect_model.py checkpoints/final_models/model_xxx.pth
```

---

## 核心算法

### 老化进化算法 (Aging Evolution)

```
Algorithm: Aging Evolution
1. 初始化: 创建 P 个随机个体填充种群队列
2. 重复 MAX_GEN 次:
   a. 锦标赛选择: 从队列随机采样 S 个，选取最优 2 个作为父代
   b. 交叉: 以概率 P_c 进行 Unit 级别交叉
   c. 变异: 以概率 P_m 进行多种变异操作
   d. 评估: 计算子代的 NTK 条件数 (fitness = -条件数)
   e. 更新: 新个体入队尾，最老个体出队首
3. 多阶段筛选与完整训练
```

### 网络编码策略

使用变长整数列表编码网络架构：

```
[unit_num, block_num_1, ..., block_num_n, 
 block_1_params..., block_2_params..., ...]
```

每个 Block 包含 5 个参数:
- `out_channels`: 输出通道数
- `groups`: 分组卷积的组数
- `pool_type`: 池化类型 (0=MaxPool, 1=AvgPool)
- `pool_stride`: 池化步长 (1 或 2)
- `has_senet`: 是否使用 SENet 注意力

### 搜索空间

| 参数 | 范围 |
|------|------|
| Unit 数量 | 3-5 |
| 每 Unit Block 数量 | 2-5 |
| 通道数 | [4, 8, 16, 32, 64] |
| 分组数 | [1, 2, 4, 8, 16, 32, 64] |
| 池化类型 | [MaxPool, AvgPool] |
| 池化步长 | [1, 2] |
| SENet | [是, 否] |

---

## 配置参数说明

主要配置位于 `src/configuration/config.py`:

### 进化算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POPULATION_SIZE` | 100 | 种群大小 (队列容量) |
| `MAX_GEN` | 5000 | 总评估个体数 |
| `TOURNAMENT_SIZE` | 5 | 锦标赛选择样本数 |
| `PROB_CROSSOVER` | 0.5 | 交叉概率 |
| `PROB_MUTATION` | 0.5 | 变异概率 |

### 筛选与训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HISTORY_TOP_N1` | 20 | 第一轮 NTK 筛选数量 |
| `SHORT_TRAIN_EPOCHS` | 20 | 短期训练轮数 |
| `HISTORY_TOP_N2` | 5 | 第二轮筛选数量 |
| `FULL_TRAIN_EPOCHS` | 300 | 完整训练轮数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 128 | 批次大小 |
| `LEARNING_RATE` | 0.1 | 初始学习率 |
| `MOMENTUM` | 0.9 | SGD 动量 |
| `WEIGHT_DECAY` | 5e-4 | 权重衰减 |

---

## 网络架构

### 基本组件

1. **ConvUnit**: 初始卷积层 (Conv-BN-ReLU)
2. **RegBlock**: 类 ResNet 残差块
   - 1x1 Conv → 3x3 GroupConv → Pool → 1x1 Conv + Shortcut
   - 可选 SENet 注意力模块
3. **RegUnit**: 由多个 RegBlock 组成
4. **SearchedNetwork**: 完整的搜索出的网络

### 网络流程

```
Input → ConvUnit → RegUnit_1 → ... → RegUnit_N → GlobalAvgPool → FC → Output
```

---

## 输出文件

### Checkpoint 文件 (`.pkl`)

```python
{
    'population': [...],  # 当前种群
    'history': [...],     # 历史所有个体
}
```

### 模型文件 (`.pth`)

```python
{
    'state_dict': ...,    # 模型权重
    'encoding': [...],    # 架构编码
    'accuracy': float,    # 验证集准确率
    'param_count': int,   # 参数量
    'history': [...],     # 训练历史
}
```

---

## 测试

### 运行所有测试

```bash
# 从项目根目录运行
cd c:\Users\gemaymini\Desktop\nas
python test\run_tests.py
```

### 运行特定测试模块

```bash
# 运行编码模块测试
python test\test_encoding.py

# 运行网络模块测试
python test\test_network.py
```

### 列出所有测试模块

```bash
python test\run_tests.py --list
```

### 测试覆盖模块

| 测试文件 | 覆盖模块 |
|----------|----------|
| test_config.py | 配置模块 (Config) |
| test_encoding.py | 编码模块 (BlockParams, Individual, Encoder) |
| test_search_space.py | 搜索空间模块 (SearchSpace, PopulationInitializer) |
| test_network.py | 网络构建模块 (SEBlock, ConvUnit, RegBlock, RegUnit, SearchedNetwork, NetworkBuilder) |
| test_mutation.py | 变异算子模块 (MutationOperator, SelectionOperator, CrossoverOperator) |
| test_dataset.py | 数据集模块 (DatasetLoader) |
| test_trainer.py | 训练器模块 (NetworkTrainer) |
| test_evaluator.py | 评估器模块 (NTKEvaluator, FitnessEvaluator) |
| test_logger.py | 日志模块 (Logger, TBLogger, FailedLogger) |
| test_evolution.py | 进化算法模块 (AgingEvolutionNAS) |

---

## TensorBoard 可视化

```bash
tensorboard --logdir=runs
```

可查看:
- 每代最佳/平均 fitness
- 种群大小变化
- Unit 数量分布

---

## 性能参考

基于 CIFAR-10 数据集的搜索结果:

| 模型 | 参数量 | 准确率 |
|------|--------|--------|
| model_3741 | - | 88.54% |
| model_2776 | - | 85.97% |
| model_3826 | - | 86.90% |

---

## 许可证

MIT License

---

## 参考文献

1. Real, E., et al. "Regularized Evolution for Image Classifier Architecture Search." AAAI 2019.
2. Jacot, A., et al. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." NeurIPS 2018.
3. Chen, W., et al. "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective." ICLR 2021.
