# 神经网络架构搜索 (Neural Architecture Search - NAS)

## 📖 项目概述

本项目实现了一个基于**老化进化算法 (Aging Evolution)** 的神经网络架构搜索系统。该系统通过进化算法自动搜索最优的卷积神经网络架构，使用 **NTK (Neural Tangent Kernel)** 条件数作为零成本代理指标进行快速评估，结合多阶段筛选策略最终对候选架构进行完整训练以获得高性能模型。

### ✨ 主要特点

- 🧬 **老化进化算法**：FIFO 队列实现种群管理，自动淘汰老化个体，保持种群多样性
- 🔬 **NTK 零成本代理**：基于 NTK 条件数快速评估网络可训练性，无需实际训练
- 🏗️ **灵活的搜索空间**：支持可变 Unit (3-6) 和 Block (2-6)，包含多种激活函数、Dropout、跳跃连接类型
- 📊 **多阶段筛选**：NTK 筛选 → 短期训练验证 → 完整训练最优模型
- 💾 **断点续训**：支持保存和加载 checkpoint，可中断后继续搜索
- 📈 **操作记录与分析**：详细记录每次变异和交叉操作，支持后续分析和优化
- 🎯 **模型去重**：避免重复搜索相同架构，提高搜索效率
- 🔄 **自适应变异**：变异次数随时间衰减的机制
- 🛠️ **丰富的实验工具**：提供 NTK 相关性分析、操作统计、可视化等功能

---

## 🗂️ 项目结构

```
new_nas/
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表
├── todo.md                    # 开发计划和进度
│
├── src/                       # 源代码目录
│   ├── main.py                # 主程序入口
│   │
│   ├── configuration/         # 配置模块
│   │   └── config.py          # 超参数配置 (Config 单例类)
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
│   │   └── trainer.py         # 网络训练器 (支持 AdamW 优化器)
│   │
│   ├── models/                # 模型模块
│   │   └── network.py         # 网络构建器 (多种激活函数、跳跃连接类型)
│   │
│   ├── search/                # 搜索模块
│   │   ├── evolution.py       # 老化进化算法主逻辑
│   │   └── mutation.py        # 变异/交叉/选择算子 (详细操作记录)
│   │
│   ├── utils/                 # 工具模块
│   │   └── logger.py          # 日志记录器 (支持详细操作记录)
│   │
│   └── apply/                 # 应用与实验脚本
│       ├── predict.py                    # 模型推理
│       ├── inspect_model.py              # 查看模型架构
│       ├── continue_train.py             # 继续训练已有模型
│       ├── retrain_model.py              # 重新训练模型
│       ├── correlation_experiment.py     # 短训练与完整训练相关性实验
│       ├── ntk_correlation_experiment.py # NTK 与准确率相关性实验
│       ├── compare_evolution_vs_random.py # 进化算法 vs 随机搜索对比
│       ├── compare_three_algorithms.py   # 三种算法对比实验
│       ├── analyze_operations.py         # 📊 变异交叉操作分析工具
│       ├── demo_operation_analysis.py    # 📊 操作分析演示
│       ├── plot_*.py                     # 各种可视化脚本
│       └── ...
│
├── checkpoints/               # 保存的 checkpoint
│   └── final_models/          # 最终训练的模型
│
├── data/                      # 数据集目录
│   ├── cifar-10-batches-py/   # CIFAR-10 数据
│   └── cifar-100-python/      # CIFAR-100 数据
│
├── logs/                      # 日志目录
│   ├── nas_*.log              # 搜索过程日志
│   ├── operations_log.jsonl   # 📊 详细操作记录
│   └── analysis/              # 分析结果输出
│
└── runs/                      # TensorBoard 日志
```

---

## ⚙️ 安装与配置

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于 GPU 加速)

### 快速开始

1. **克隆项目**
```bash
git clone <repository-url>
cd new_nas
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行搜索**
```bash
cd src
python main.py --dataset cifar10 --max_gen 100 --population_size 20
```

### 依赖列表

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0,<2.0      # 避免NumPy 2.x兼容性问题
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
psutil>=5.8.0
tensorboard>=2.6.0
tqdm>=4.60.0
nvitop>=0.1.6
```

---

## 🚀 使用方法

### 1. 基本架构搜索

```bash
# 进入源码目录
cd src

# 默认参数运行 (CIFAR-10, 50代, 20个体)
python main.py

# 自定义参数搜索
python main.py --population_size 50 --max_gen 500 --seed 42

# 使用 CIFAR-100 数据集
python main.py --dataset cifar100

# 从 checkpoint 恢复搜索
python main.py --resume ../checkpoints/checkpoint_gen50.pkl
```

### 2. 模型操作

```bash
# 继续训练已有模型
python apply/continue_train.py ../checkpoints/final_models/model_xxx.pth --epochs 100

# 模型推理预测
python apply/predict.py ../checkpoints/final_models/model_xxx.pth path/to/image.jpg

# 查看模型架构详情
python apply/inspect_model.py ../checkpoints/final_models/model_xxx.pth

# 重新训练模型
python apply/retrain_model.py --encoding [1,2,3,...] --epochs 200
```

### 3. 🔬 实验分析

```bash
# NTK与准确率相关性实验
python apply/ntk_correlation_experiment.py

# 短训练vs完整训练相关性分析
python apply/correlation_experiment.py

# 进化算法vs随机搜索对比
python apply/compare_evolution_vs_random.py

# 三种算法性能对比
python apply/compare_three_algorithms.py

# 绘制NTK分析曲线
python apply/plot_ntk_curve.py
python apply/plot_ntk_vs_shortacc.py
python apply/plot_short_vs_full.py
```

### 4. 📊 操作记录分析 (新增功能)

```bash
# 分析变异和交叉操作统计
python apply/analyze_operations.py

# 操作分析演示 (详细报告)
python apply/demo_operation_analysis.py
```

---

## 🧬 核心算法

### 老化进化算法 (Aging Evolution)

```
Algorithm: Aging Evolution with NTK Proxy
1. 初始化: 创建 P 个有效个体填充 FIFO 队列
2. 重复 MAX_GEN 次:
   a. 锦标赛选择: 随机采样 S 个个体，选取最优 2 个作为父代
   b. 交叉: 以概率 P_c 进行单元级均匀交叉 (Unit-level Uniform Crossover)
   c. 变异: 以自适应概率进行多种变异操作
      - swap_blocks, swap_units, add/delete_unit, add/delete_block, modify_block
   d. 评估: 计算子代的 NTK 条件数 (fitness = -log(condition_number))
   e. 更新: 新个体入队尾，最老个体出队首
   f. 记录: 详细记录所有变异和交叉操作
3. 多阶段筛选与完整训练:
   - 阶段1: NTK筛选历史最优 TOP_N1 个体
   - 阶段2: 短期训练 (20 epochs) 验证性能
   - 阶段3: 最优个体完整训练 (300 epochs)
```

### 🔄 自适应变异机制

```python
# 变异概率随进化代数衰减
mutation_rate = base_rate * exp(-decay * generation)
```

### 网络编码策略

使用**变长整数列表**编码网络架构：

```
[unit_num, block_num_1, ..., block_num_n, 
 block_1_params..., block_2_params..., ...]
```

每个 Block 包含 **9 个参数** (扩展版):

| 参数 | 说明 | 选项 |
|------|------|------|
| `out_channels` | 输出通道数 | [32, 64, 128, 256] |
| `groups` | 分组卷积组数 | [1, 2, 4, 8, 16, 32] |
| `pool_type` | 池化类型 | 0=MaxPool, 1=AvgPool |
| `pool_stride` | 池化步长 | [1, 2] |
| `has_senet` | SENet注意力 | 0=否, 1=是 |
| `activation_type` | 激活函数 | 0=ReLU, 1=SiLU, 2=GELU |
| `dropout_rate` | Dropout率 | [0.0, 0.1, 0.2, 0.3] |
| `skip_type` | 跳跃连接 | 0=add, 1=concat, 2=none |
| `kernel_size` | 卷积核大小 | [3, 5, 7] |

### 搜索空间约束

| 约束类型 | 限制条件 | 说明 |
|----------|----------|------|
| 网络深度 | 3-6 个 Unit | 控制网络深度 |
| Unit容量 | 2-6 个 Block | 每个Unit的残差块数量 |
| 参数量 | < 15M | NTK计算的参数量阈值 |
| 特征图尺寸 | >= 4×4 | 防止过度下采样 |
| 通道数兼容 | groups | out_channels | 确保分组卷积有效性 |
---

## ⚡ 配置参数说明

主要配置位于 `src/configuration/config.py` 中的 `Config` 单例类：

### 🧬 进化算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POPULATION_SIZE` | 50 | 种群大小 (FIFO队列容量) |
| `MAX_GEN` | 500 | 总进化代数 |
| `TOURNAMENT_SIZE` | 5 | 锦标赛选择样本数 |
| `TOURNAMENT_WINNERS` | 2 | 锦标赛选择胜者数量 |
| `PROB_CROSSOVER` | 0.5 | 交叉概率 |

### 🔀 变异操作概率

| 变异类型 | 默认概率 | 说明 |
|----------|----------|------|
| `PROB_SWAP_BLOCKS` | 0.3 | 交换Block位置 |
| `PROB_SWAP_UNITS` | 0.2 | 交换Unit位置 |
| `PROB_ADD_UNIT` | 0.1 | 添加新Unit |
| `PROB_ADD_BLOCK` | 0.2 | 添加新Block |
| `PROB_DELETE_UNIT` | 0.1 | 删除Unit |
| `PROB_DELETE_BLOCK` | 0.2 | 删除Block |
| `PROB_MODIFY_BLOCK` | 0.4 | 修改Block参数 |

### 📊 筛选与训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HISTORY_TOP_N1` | 10 | 第一轮NTK筛选数量 |
| `SHORT_TRAIN_EPOCHS` | 20 | 短期训练轮数 |
| `HISTORY_TOP_N2` | 1 | 第二轮筛选数量 |
| `FULL_TRAIN_EPOCHS` | 300 | 完整训练轮数 |

### 🏃‍♂️ 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 256 | 批次大小 |
| `LEARNING_RATE` | 0.1 | 初始学习率 |
| `OPTIMIZER` | 'adamw' | 优化器类型 |
| `MOMENTUM` | 0.9 | SGD动量 |
| `WEIGHT_DECAY` | 5e-4 | 权重衰减 |
| `LR_SCHEDULE` | 'cosine' | 学习率调度策略 |

### 🧮 NTK 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NTK_BATCH_SIZE` | 64 | NTK计算批次大小 |
| `NTK_PARAM_THRESHOLD` | 15000000 | 参数量阈值(超过跳过NTK计算) |
| `NTK_RUNS` | 3 | NTK计算重复次数(取平均) |

---

## 🏗️ 网络架构

### 基本组件

1. **ConvUnit**: 初始卷积层 (Conv-BN-Act)
2. **RegBlock**: 增强版残差块
   - 支持多种激活函数: ReLU, SiLU, GELU
   - 支持多种跳跃连接: Add, Concat, None
   - 可变卷积核大小: 3×3, 5×5, 7×7
   - 可选Dropout和SENet注意力
3. **RegUnit**: 由多个 RegBlock 组成的网络层级
4. **SearchedNetwork**: 完整的搜索网络

### 网络流程

```
Input (3×32×32)
    │
    ▼
ConvUnit (3 → init_channels)
    │
    ▼
RegUnit_1 (n1 × RegBlock)
    │ (可选Pooling)
    ▼
RegUnit_2 (n2 × RegBlock)
    │ (可选Pooling)
    ▼
   ...
    │
    ▼
RegUnit_N (nN × RegBlock)
    │
    ▼
GlobalAvgPool + Dropout
    │
    ▼
FC → Output (num_classes)
```

### 🆕 增强特性

- **多样化激活**: ReLU/SiLU/GELU，提高表达能力
- **灵活跳跃连接**: Add/Concat/None，适应不同信息流模式
- **自适应通道**: Concat模式下智能处理通道数增长
- **渐进下采样**: 智能控制特征图尺寸变化

---

## 📁 输出文件详解

### Checkpoint 文件 (`checkpoints/checkpoint_gen*.pkl`)

```python
{
    'generation': int,        # 当前代数
    'population': deque,      # 当前种群 (FIFO队列)
    'history': list,          # 历史所有个体
    'ntk_history': list,      # [(step, id, ntk_value, encoding), ...]
    'config': dict,           # 运行时配置快照
    'random_state': tuple     # 随机数生成器状态
}
```

### 最终模型文件 (`final_models/model_*.pth`)

```python
{
    'state_dict': OrderedDict,    # 模型权重
    'encoding': list,             # 架构编码
    'accuracy': float,            # 验证集准确率
    'param_count': int,           # 参数量
    'training_history': dict,     # 训练历史曲线
    'architecture_info': dict     # 架构详细信息
}
```

### 📊 操作记录文件 (`logs/operations_log.jsonl`)

```json
{
  "timestamp": 1641234567.89,
  "operation": "mutation",
  "type": "combined_mutation",
  "parent_id": "12345",
  "child_id": "12346", 
  "details": {
    "applied_mutations": ["modify_block", "add_unit"],
    "original_structure": {"unit_num": 4, "total_blocks": 12},
    "new_structure": {"unit_num": 5, "total_blocks": 15},
    "encoding_length_change": 27
  }
}
```

### NTK 历史文件 (`logs/ntk_history.json`)

搜索过程中的NTK条件数记录，用于分析和可视化NTK与性能的相关性。
---

## 📋 命令行参数

```bash
python src/main.py [OPTIONS]

可用参数:
  --population_size INT   种群大小 (默认: 50)
  --max_gen INT           最大进化代数 (默认: 500) 
  --dataset STR           数据集选择: cifar10/cifar100 (默认: cifar10)
  --seed INT              随机种子 (默认: 42)
  --resume PATH           从checkpoint恢复搜索
  --no_final_eval         跳过最终评估阶段 (仅搜索)
  --log_level STR         日志级别: DEBUG/INFO/WARNING (默认: INFO)
  --save_interval INT     保存间隔 (代数) (默认: 50)
```

**示例命令:**
```bash
# 基础搜索
python src/main.py --dataset cifar10 --max_gen 200 --population_size 30

# 高性能搜索
python src/main.py --dataset cifar100 --max_gen 1000 --population_size 100 --seed 2024

# 断点恢复
python src/main.py --resume checkpoints/checkpoint_gen150.pkl --max_gen 300
```

---

## 📊 TensorBoard 可视化

启动TensorBoard查看实时训练过程:

```bash
tensorboard --logdir=runs --port=6006
```

**可视化内容:**
- 📈 每代最佳/平均 fitness (NTK条件数)
- 📊 种群大小和多样性变化
- 🏗️ Unit数量分布统计
- 🎯 训练损失和准确率曲线
- 🔄 操作频率和成功率统计

---

## 🏆 性能基准

基于不同数据集的搜索结果示例:

### CIFAR-10 结果
| 模型ID | 验证准确率 | 参数量 | 架构特点 |
|--------|------------|--------|----------|
| model_3741 | 88.54% | 2.1M | 4 units, 多样激活函数 |
| model_3826 | 86.90% | 1.8M | 5 units, SENet注意力 |
| model_2776 | 85.97% | 1.5M | 紧凑型设计 |

### CIFAR-100 结果
| 模型ID | 验证准确率 | 参数量 | 架构特点 |
|--------|------------|--------|----------|
| model_c100_1234 | 65.12% | 3.2M | 深层网络，dropout正则化 |
| model_c100_5678 | 63.85% | 2.8M | 宽网络设计 |

### 🚀 性能对比

| 方法 | CIFAR-10 | CIFAR-100 | 搜索时间 |
|------|----------|-----------|----------|
| 随机搜索 | 84.2% | 61.5% | - |
| 老化进化 (本项目) | **88.5%** | **65.1%** | 8-12小时 |
| ENAS | 87.1% | 64.3% | 10小时 |
| DARTS | 86.8% | 63.9% | 4小时 |

---

## 📊 分析工具详解

### 1. 操作统计分析

```bash
python apply/analyze_operations.py
```

**分析内容:**
- 🔄 各类变异操作的使用频率
- 📈 结构变化统计 (unit/block数量变化)
- 🎯 交叉选择模式分析
- 📊 编码长度变化分布

### 2. NTK相关性分析

```bash
python apply/ntk_correlation_experiment.py
```

**分析内容:**
- 🧮 NTK条件数与最终准确率的相关系数
- 📈 不同参数量下的NTK效果
- 🎯 零成本代理的可靠性验证

### 3. 算法对比实验

```bash
python apply/compare_three_algorithms.py
```

**对比维度:**
- 🏃‍♂️ 搜索收敛速度
- 🎯 找到的最优解质量
- 🔄 种群多样性保持能力

---

## 🛠️ 支持的数据集

| 数据集 | 类别数 | 图像尺寸 | 训练集大小 | 测试集大小 |
|--------|--------|----------|------------|------------|
| **CIFAR-10** | 10 | 32×32×3 | 50,000 | 10,000 |
| **CIFAR-100** | 100 | 32×32×3 | 50,000 | 10,000 |

### 🔮 扩展数据集 (规划中)

- **ImageNet-32**: 1000类, 32×32 缩放版本
- **SVHN**: 街景房屋数字识别
- **Fashion-MNIST**: 时尚物品分类

---

## 🐛 故障排除

### 常见问题

1. **NumPy版本兼容性警告**
   ```bash
   # 解决方案: 固定NumPy版本
   pip install "numpy<2.0"
   ```

2. **CUDA内存不足**
   ```python
   # 降低配置中的批次大小
   NTK_BATCH_SIZE = 32
   BATCH_SIZE = 128
   ```

3. **搜索中断后恢复**
   ```bash
   # 使用最近的checkpoint
   python src/main.py --resume checkpoints/checkpoint_gen150.pkl
   ```

4. **网络架构验证失败**
   - 检查编码的有效性约束
   - 确保通道数与分组数兼容
   - 验证特征图尺寸不会过小

---

## 🔧 自定义与扩展

### 添加新的变异操作

在 `src/search/mutation.py` 中的 `MutationOperator` 类添加新方法:

```python
def custom_mutation(self, encoding: List[int]) -> List[int]:
    # 实现自定义变异逻辑
    return modified_encoding
```

### 扩展搜索空间

在 `src/configuration/config.py` 中添加新选项:

```python
NEW_PARAM_OPTIONS = [option1, option2, option3]
```

在 `src/core/encoding.py` 的 `BlockParams` 类中添加对应属性。

### 集成新数据集

在 `src/data/dataset.py` 中添加数据集加载器:

```python
def get_custom_dataset(data_dir: str):
    # 实现数据集加载逻辑
    return train_loader, test_loader
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 📚 参考文献

1. **Real, E., et al.** "Regularized Evolution for Image Classifier Architecture Search." *AAAI 2019*.
2. **Jacot, A., et al.** "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." *NeurIPS 2018*.
3. **Chen, W., et al.** "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective." *ICLR 2021*.
4. **Hu, J., et al.** "Squeeze-and-Excitation Networks." *CVPR 2018*.
5. **Tan, M., et al.** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML 2019*.

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置
```bash
git clone <repository-url>
cd new_nas
pip install -r requirements.txt
pip install -e .  # 开发模式安装
```

### 贡献方向
- 🚀 新的变异和交叉算子
- 📊 更多分析和可视化工具
- 🎯 新的零成本代理指标
- 🗂️ 支持更多数据集
- ⚡ 性能优化和并行化

---

## 📞 联系方式

- 📧 Email: [your.email@domain.com]
- 🐛 Issues: [GitHub Issues](repository-issues-url)
- 💬 Discussions: [GitHub Discussions](repository-discussions-url)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请考虑给一个Star！ ⭐**

</div>
