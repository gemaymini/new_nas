# 神经网络架构搜索 (Neural Architecture Search - NAS)

## 项目概述

本项目实现了一个基于**老化进化算法 (Aging Evolution)** 的神经网络架构搜索系统。该系统通过进化算法自动搜索最优的卷积神经网络架构，使用 **NTK (Neural Tangent Kernel)** 条件数作为零成本代理指标进行快速评估（无需完整训练），结合多阶段筛选策略最终对候选架构进行完整训练以获得高性能模型。

### 主要特点

- 🧬 **老化进化算法**：使用 FIFO 队列实现种群管理，自动淘汰老化个体，保持种群多样性
- 🔬 **NTK 零成本代理**：基于 NTK 条件数快速评估网络可训练性，无需实际训练即可筛选候选架构
- 🏗️ **灵活的搜索空间**：支持可变数量的 Unit (3-6) 和 Block (2-6)，包含 SENet 注意力机制和分组卷积
- 📊 **多阶段筛选**：NTK 筛选 Top-N1 → 短期训练验证 → 完整训练最优模型
- 💾 **断点续训**：支持保存和加载 checkpoint，可中断后继续搜索
- 📈 **实验分析工具**：提供 NTK 相关性分析、训练曲线绘制等实验脚本

---

## 项目结构

```
new_nas/
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表
│
├── src/                       # 源代码目录
│   ├── main.py                # 主程序入口
│   │
│   ├── configuration/         # 配置模块
│   │   └── config.py          # 超参数配置
│   │
│   ├── core/                  # 核心模块
│   │   ├── encoding.py        # 编码器与个体类 (BlockParams, Individual, Encoder)
│   │   └── search_space.py    # 搜索空间定义与种群初始化
│   │
│   ├── data/                  # 数据模块
│   │   └── dataset.py         # 数据集加载器 (CIFAR-10/100)
│   │
│   ├── engine/                # 引擎模块
│   │   ├── evaluator.py       # NTK 评估器 & 最终评估器
│   │   └── trainer.py         # 网络训练器
│   │
│   ├── models/                # 模型模块
│   │   └── network.py         # 网络构建器 (SEBlock, ConvUnit, RegBlock, RegUnit)
│   │
│   ├── search/                # 搜索模块
│   │   ├── evolution.py       # 老化进化算法主逻辑
│   │   └── mutation.py        # 变异/交叉/选择算子
│   │
│   ├── utils/                 # 工具模块
│   │   └── logger.py          # 日志记录器
│   │
│   └── apply/                 # 应用与实验脚本
│       ├── predict.py                    # 模型推理
│       ├── inspect_model.py              # 查看模型架构
│       ├── continue_train.py             # 继续训练已有模型
│       ├── retrain_model.py              # 重新训练模型
│       ├── correlation_experiment.py     # 短训练与完整训练相关性实验
│       ├── ntk_correlation_experiment.py # NTK 与准确率相关性实验
│       ├── plot_ntk_curve.py             # 绘制 NTK 曲线
│       ├── plot_ntk_vs_shortacc.py       # NTK vs 短训练准确率
│       └── plot_short_vs_full.py         # 短训练 vs 完整训练准确率
│
├── checkpoints/               # 保存的 checkpoint
│   └── final_models/          # 最终训练的模型
│
├── data/                      # 数据集目录
│   ├── cifar-10-batches-py/   # CIFAR-10 数据
│   └── cifar-100-python/      # CIFAR-100 数据
│
├── logs/                      # 日志目录
└── runs/                      # TensorBoard 日志
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
nvitop>=0.1.6
```

---

## 使用方法

### 1. 运行架构搜索

```bash
# 进入 src 目录
cd src

# 默认参数运行 (CIFAR-10)
python main.py

# 自定义参数
python main.py --population_size 50 --max_gen 500 --seed 42

# 使用 CIFAR-100 数据集
python main.py --dataset cifar100

# 从 checkpoint 恢复搜索
python main.py --resume ../checkpoints/checkpoint_step100.pkl
```

### 2. 继续训练已有模型

```bash
python apply/continue_train.py ../checkpoints/final_models/model_xxx.pth --epochs 100 --lr 0.01
```

### 3. 模型推理

```bash
python apply/predict.py ../checkpoints/final_models/model_xxx.pth path/to/image.jpg
```

### 4. 查看模型架构

```bash
python apply/inspect_model.py ../checkpoints/final_models/model_xxx.pth
```

### 5. 实验分析

```bash
# NTK 相关性实验
python apply/ntk_correlation_experiment.py

# 短训练 vs 完整训练相关性分析
python apply/correlation_experiment.py

# 绘制 NTK 曲线
python apply/plot_ntk_curve.py
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

| 参数 | 范围 | 说明 |
|------|------|------|
| Unit 数量 | 3-6 | 网络深度层级 |
| 每 Unit Block 数量 | 2-6 | 每个层级的残差块数量 |
| 通道数 | [32, 64, 128, 256, 512] | 中间层通道数 |
| 分组数 | [1, 2, 4, 8, 16, 32] | 分组卷积的组数 |
| 池化类型 | [MaxPool, AvgPool] | 下采样方式 |
| 池化步长 | [1, 2] | 空间分辨率变化 |
| SENet | [是, 否] | 是否使用注意力机制 |
| 通道扩展系数 | 2 | 输出通道 = 中间通道 × 2 |

---

## 配置参数说明

主要配置位于 `src/configuration/config.py`:

### 进化算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POPULATION_SIZE` | 50 | 种群大小 (队列容量) |
| `MAX_GEN` | 500 | 总进化代数 |
| `TOURNAMENT_SIZE` | 5 | 锦标赛选择样本数 |
| `TOURNAMENT_WINNERS` | 2 | 锦标赛选择胜者数量 |
| `PROB_CROSSOVER` | 0.5 | 交叉概率 |
| `PROB_MUTATION` | 0.5 | 变异概率 |

### 筛选与训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HISTORY_TOP_N1` | 10 | 第一轮 NTK 筛选数量 |
| `SHORT_TRAIN_EPOCHS` | 20 | 短期训练轮数 |
| `HISTORY_TOP_N2` | 1 | 第二轮筛选数量 |
| `FULL_TRAIN_EPOCHS` | 300 | 完整训练轮数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 256 | 批次大小 |
| `LEARNING_RATE` | 0.1 | 初始学习率 |
| `MOMENTUM` | 0.9 | SGD 动量 |
| `WEIGHT_DECAY` | 5e-4 | 权重衰减 |

### NTK 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NTK_BATCH_SIZE` | 64 | NTK 计算批次大小 |
| `NTK_PARAM_THRESHOLD` | 15000000 | 参数量阈值（超过跳过NTK计算） |

---

## 网络架构

### 基本组件

1. **ConvUnit**: 初始卷积层 (Conv-BN-ReLU)
2. **RegBlock**: 类 ResNet/ResNeXt 残差块
   - 1x1 Conv → 3x3 GroupConv → Pool → 1x1 Conv + Shortcut
   - 输出通道 = 中间通道 × EXPANSION (默认2)
   - 可选 SENet 注意力模块
3. **RegUnit**: 由多个 RegBlock 组成的网络层级
4. **SearchedNetwork**: 完整的搜索出的网络

### 网络流程

```
Input (3×32×32)
    │
    ▼
ConvUnit (3 → 64 channels)
    │
    ▼
RegUnit_1 (多个 RegBlock)
    │
    ▼
RegUnit_2 (多个 RegBlock)
    │
    ▼
   ...
    │
    ▼
RegUnit_N (多个 RegBlock)
    │
    ▼
GlobalAvgPool
    │
    ▼
FC → Output (10/100 classes)
```

---

## 输出文件

### Checkpoint 文件 (`.pkl`)

```python
{
    'population': [...],      # 当前种群 (deque)
    'history': [...],         # 历史所有个体
    'ntk_history': [...],     # NTK 历史记录 [(step, id, ntk_value, encoding), ...]
}
```

### 模型文件 (`.pth`)

```python
{
    'state_dict': ...,        # 模型权重
    'encoding': [...],        # 架构编码
    'accuracy': float,        # 验证集准确率
    'param_count': int,       # 参数量
    'history': [...],         # 训练历史
}
```

### NTK 历史文件 (`ntk_history.json`)

搜索过程中的 NTK 条件数记录，用于分析和可视化。

---

## 命令行参数

```bash
python main.py [OPTIONS]

参数说明:
  --population_size INT   种群大小 (默认: 50)
  --max_gen INT           最大进化代数 (默认: 500)
  --dataset STR           数据集 cifar10/cifar100 (默认: cifar10)
  --seed INT              随机种子 (默认: 42)
  --resume PATH           从 checkpoint 恢复搜索
  --no_final_eval         跳过最终评估阶段
```

---

## TensorBoard 可视化

```bash
tensorboard --logdir=runs
```

可查看:
- 每代最佳/平均 fitness (NTK 条件数)
- 种群大小变化
- Unit 数量分布
- 训练损失和准确率曲线

---

## 性能参考

基于 CIFAR-10 数据集的搜索结果:

| 模型 | 准确率 |
|------|--------|
| model_3741 | 88.54% |
| model_3826 | 86.90% |
| model_2776 | 85.97% |

---

## 支持的数据集

| 数据集 | 类别数 | 图像大小 |
|--------|--------|----------|
| CIFAR-10 | 10 | 32×32 |
| CIFAR-100 | 100 | 32×32 |

---

## 许可证

MIT License

---

## 参考文献

1. Real, E., et al. "Regularized Evolution for Image Classifier Architecture Search." AAAI 2019.
2. Jacot, A., et al. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." NeurIPS 2018.
3. Chen, W., et al. "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective." ICLR 2021.
4. Hu, J., et al. "Squeeze-and-Excitation Networks." CVPR 2018.
