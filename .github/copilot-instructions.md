# NAS 项目 Copilot 指南

## 项目概述
基于**老化进化算法 (Aging Evolution)** 的神经网络架构搜索系统，使用 **NTK 条件数**作为零成本代理进行架构评估。

## 核心架构

### 数据流
```
main.py → AgingEvolutionNAS.run_search() → fitness_evaluator(NTK) → run_screening_and_training()
           ↓                                                            ↓
    population_initializer.create_valid_individual()              短期训练筛选 → 完整训练
           ↓
    mutation_operator / crossover_operator
```

### 关键模块职责
| 模块 | 职责 |
|------|------|
| [src/configuration/config.py](src/configuration/config.py) | 所有超参数配置 (Config 单例类) |
| [src/core/encoding.py](src/core/encoding.py) | 架构编码：`Individual`、`BlockParams`、`Encoder` |
| [src/search/evolution.py](src/search/evolution.py) | 老化进化主逻辑：`AgingEvolutionNAS` |
| [src/engine/evaluator.py](src/engine/evaluator.py) | NTK 评估：`NTKEvaluator`、`FinalEvaluator` |
| [src/models/network.py](src/models/network.py) | 网络构建：`RegBlock`、`RegUnit`、`NetworkBuilder` |

## 架构编码规范

### 编码结构 (变长整数列表)
```
[unit_num, block_num_1, ..., block_num_n, block_params_1, ..., block_params_m]
```

### BlockParams (9 个参数)
```python
[out_channels, groups, pool_type, pool_stride, has_senet, 
 activation_type, dropout_rate*100, skip_type, kernel_size]
```
- `activation_type`: 0=ReLU, 1=SiLU, 2=GELU
- `skip_type`: 0=add, 1=concat, 2=none
- `dropout_rate`: 存储时乘 100 转为整数

## 开发命令

```bash
# 运行搜索 (从 src 目录)
python src/main.py --dataset cifar10 --max_gen 100 --population_size 20

# 断点续训
python src/main.py --resume checkpoints/checkpoint_gen50.pkl

# 实验脚本 (在 src/apply 目录)
python src/apply/ntk_correlation_experiment.py
python src/apply/plot_ntk_curve.py
```

## 编码规范

### 配置修改
- 所有超参数在 `config.py` 的 `Config` 类中定义
- 运行时参数通过 `argparse` 覆盖 config 值

### 变异算子扩展
在 [src/search/mutation.py](src/search/mutation.py) 的 `MutationOperator` 类中添加，并在 `mutate()` 方法中集成：
```python
def mutate(self, individual: Individual) -> Individual:
    # 按概率依次应用各变异算子
```

### 搜索空间扩展
1. 在 `config.py` 添加新选项列表 (如 `XXX_OPTIONS = [...]`)
2. 在 `BlockParams` 类添加对应属性
3. 更新 `BLOCK_PARAM_COUNT` 常量
4. 在 `search_space.py` 添加采样方法

### 网络组件
- 所有卷积块继承 `nn.Module`，实现 `forward()` 方法
- `RegBlock` 支持可变卷积核、激活函数、跳跃连接类型
- `EXPANSION` 参数控制通道扩展比 (类似 ResNeXt)

## 关键约束

### NTK 评估
- 参数量超过 `NTK_PARAM_THRESHOLD` 的模型跳过评估，返回高惩罚值
- 使用 `train_mode=True` 计算 NTK (与原始论文一致)
- 多次运行取平均以稳定结果

### 编码验证
- `groups` 必须能整除 `out_channels`
- unit 数量范围: `[MIN_UNIT_NUM, MAX_UNIT_NUM]`
- block 数量范围: `[MIN_BLOCK_NUM, MAX_BLOCK_NUM]`
- 使用 `Encoder.validate_encoding()` 验证

### GPU 内存管理
- 评估后调用 `clear_gpu_memory()` 清理缓存
- 已移除 512 通道选项以避免 OOM

## 日志与输出
- 训练日志: `logs/nas_*.log`
- TensorBoard: `runs/`
- NTK 历史: `logs/ntk_history.json`
- Checkpoints: `checkpoints/`
