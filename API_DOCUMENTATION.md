# NAS Project API 文档

本文档详细说明了项目中主要模块、类和函数的输入参数、输出参数及功能作用。

## 1. Core Modules (核心模块)

### `src/core/encoding.py`

#### `BlockParams` 类
**作用**: 用于存储和管理单个网络块（Block）的超参数。

*   **`__init__`**
    *   **输入参数**:
        *   `out_channels` (int): 输出通道数。
        *   `groups` (int): 卷积分组数。
        *   `pool_type` (int): 池化类型 (0=MaxPool, 1=AvgPool)。
        *   `pool_stride` (int): 池化步长 (1 或 2)。
        *   `has_senet` (int): 是否使用 SE 模块 (1=是, 0=否)。
        *   `activation_type` (int, optional): 激活函数类型 (0=ReLU, 1=SiLU)。默认 0。
        *   `dropout_rate` (float, optional): Dropout 率。默认 0.0。
        *   `skip_type` (int, optional): 跳跃连接类型 (0=add, 1=concat, 2=none)。默认 0。
        *   `kernel_size` (int, optional): 卷积核大小 (3 或 5)。默认 3。
        *   `expansion` (int, optional): 中间通道膨胀系数 (1 或 2)。默认 2。
    *   **输出参数**: 无 (构造函数)。

*   **`to_list`**
    *   **作用**: 将块参数序列化为整数列表。
    *   **输入参数**: 无。
    *   **输出参数**: `List[int]` (扁平化的参数列表)。

*   **`from_list` (类方法)**
    *   **作用**: 从整数列表反序列化为 `BlockParams` 对象。
    *   **输入参数**: `params` (List): 包含参数的列表。
    *   **输出参数**: `BlockParams` 对象。

#### `Individual` 类
**作用**: 表示一个候选架构个体。

*   **`__init__`**
    *   **输入参数**: `encoding` (List[int], optional): 架构的基因编码。
    *   **输出参数**: 无。

*   **`copy`**
    *   **作用**: 创建个体的深拷贝。
    *   **输入参数**: 无。
    *   **输出参数**: `Individual` (新的个体及其副本数据)。

#### `Encoder` 类
**作用**: 处理变长架构编码的编码与解码。

*   **`decode` (静态方法)**
    *   **作用**: 将整数列表编码解析为结构化组件。
    *   **输入参数**: `encoding` (List[int]): 架构编码。
    *   **输出参数**: `Tuple[int, List[int], List[List[BlockParams]]]`
        *   Unit 数量
        *   每个 Unit 的 Block 数量列表
        *   嵌套的 `BlockParams` 对象列表

*   **`encode` (静态方法)**
    *   **作用**: 将结构化组件打包为整数列表编码。
    *   **输入参数**:
        *   `unit_num` (int): Unit 数量。
        *   `block_nums` (List[int]): 每个 Unit 的 Block 数量。
        *   `block_params_list` (List[List[BlockParams]]): Block 参数列表。
    *   **输出参数**: `List[int]` (扁平化编码)。

*   **`validate_encoding` (静态方法)**
    *   **作用**: 验证编码是否符合搜索空间约束（如通道数限制、特征图大小限制等）。
    *   **输入参数**: `encoding` (List[int])。
    *   **输出参数**: `bool` (True 表示合法，False 表示非法)。

### `src/core/search_space.py`

#### `SearchSpace` 类
**作用**: 定义超参数的可选范围及随机采样方法。

*   **采样方法 (如 `sample_unit_num`, `sample_block_num`, `sample_channel` 等)**
    *   **作用**: 从预定义配置(`config.py`)中随机采样一个对应参数的值。
    *   **输入参数**: 无 (部分如 `sample_skip_type` 接受 `allow_concat` bool 参数)。
    *   **输出参数**: 对应类型的随机值 (int 或 float)。

*   **`sample_block_params`**
    *   **作用**: 随机生成一个完整 Block 的参数对象。
    *   **输入参数**: `allow_concat` (bool): 是否允许采样 Concat 类型的跳跃连接。
    *   **输出参数**: `BlockParams` 对象。

#### `PopulationInitializer` 类
**作用**: 生成满足约束的初始种群个体。

*   **`create_valid_individual`**
    *   **作用**: 生成一个通过所有验证检查（编码结构、参数量限制等）的合法个体。
    *   **输入参数**: 无。
    *   **输出参数**: `Individual` (包含合法编码)。

---

## 2. Models (模型定义)

### `src/models/network.py`

#### `SearchedNetwork` 类
**作用**: 根据基因编码构建的 PyTorch 神经网络模型。

*   **`__init__`**
    *   **输入参数**:
        *   `encoding` (List[int]): 架构编码。
        *   `input_channels` (int): 输入图像通道数。默认 3。
        *   `num_classes` (int): 分类类别数。默认 10。
    *   **输出参数**: 无。

*   **`forward`**
    *   **作用**: 前向传播。
    *   **输入参数**: `x` (Tensor): 输入数据。
    *   **输出参数**: `Tensor`: 网络输出（Logits）。

*   **`get_param_count`**
    *   **作用**: 计算模型参数量。
    *   **输入参数**: 无。
    *   **输出参数**: `int`: 参数总数。

#### `NetworkBuilder` 类
**作用**: 构建网络的辅助工厂类。

*   **`build_from_encoding` / `build_from_individual`**
    *   **作用**: 静态方法，根据编码或个体对象创建 `SearchedNetwork` 实例。
    *   **输入参数**: 编码/个体, 输入通道, 类别数。
    *   **输出参数**: `SearchedNetwork` 实例。

---

## 3. Search & Engine (搜索与执行引擎)

### `src/search/evolution.py`

#### `AgingEvolutionNAS` 类
**作用**: 实现老化进化算法 (Regularized Evolution) 的主控制器。

*   **`initialize_population`**
    *   **作用**: 初始化随机种群。
    *   **输入参数**: 无。
    *   **输出参数**: 无 (更新内部 `self.population`)。

*   **`step`**
    *   **作用**: 执行一步进化（选择父母 -> 交叉变异 -> 评估子代 -> 更新种群）。
    *   **输入参数**: 无。
    *   **输出参数**: `bool` (是否成功执行)。

*   **`run_search`**
    *   **作用**: 运行完整的搜索循环，直到达到最大代数。
    *   **输入参数**: 无。
    *   **输出参数**: 无。

*   **`run_screening_and_training`**
    *   **作用**: 执行搜索后的筛选流程（NTK筛选 -> 短期训练筛选 -> 最终全训练）。
    *   **输入参数**: 无。
    *   **输出参数**: `Optional[Individual]` (最佳模型个体)。

### `src/engine/evaluator.py`

#### `NTKEvaluator` 类
**作用**: 计算神经正切核 (NTK) 条件数作为零成本代理指标。

*   **`compute_ntk_score`**
    *   **作用**: 计算网络的平均 NTK 条件数。
    *   **输入参数**:
        *   `network`: PyTorch 模型。
        *   `num_runs`: 重复计算次数以取平均。
    *   **输出参数**: `float` (NTK 分数)。

#### `FinalEvaluator` 类
**作用**: 负责模型的实际训练和验证（短期或长期）。

*   **`evaluate_individual`**
    *   **作用**: 构建模型并进行完整训练。
    *   **输入参数**:
        *   `individual`: 待评估个体。
        *   `epochs`: 训练轮数。
    *   **输出参数**: `Tuple[float, dict]` (最佳验证准确率, 详细结果字典)。

### `src/engine/trainer.py`

#### `NetworkTrainer` 类
**作用**: 封装通用的 PyTorch 训练循环。

*   **`train_network`**
    *   **作用**: 执行完整的训练流程（训练+验证+学习率调度）。
    *   **输入参数**:
        *   `model`: 待训练模型。
        *   `trainloader`: 训练数据迭代器。
        *   `testloader`: 验证数据迭代器。
        *   `epochs`: 训练轮数。
        *   `optimizer_name`, `lr`, 等: 优化器配置。
    *   **输出参数**: `Tuple[float, List[dict]]` (最佳测试准确率, 训练历史记录)。

---

## 4. Utils (工具类)

### `src/utils/checkpoint.py`
**作用**: 模型检查点加载与解析工具。

*   **`load_checkpoint`**
    *   **输入**: `model_path` (str), `device` (str)。
    *   **输出**: 检查点字典。
    *   **作用**: 安全加载 `.pth` 文件。

*   **`extract_encoding_from_checkpoint`**
    *   **输入**: `checkpoint` (dict)。
    *   **输出**: `List[int]` (架构编码)。
    *   **作用**: 从检查点中提取架构基因编码。

### `src/utils/constraints.py`
**作用**: 参数约束检查工具。

*   **`evaluate_encoding_params`**
    *   **输入**: `encoding` (list)。
    *   **输出**: `Tuple[bool, str, int]` (是否合法, 原因, 参数量)。
    *   **作用**: 快速估算编码对应的模型参数量并验证是否在允许范围内。

### `src/utils/generation.py`
**作用**: 生成合法子代的辅助函数。

*   **`generate_valid_child`**
    *   **输入**: 父母个体、交叉/变异/修复函数句柄、最大尝试次数等。
    *   **输出**: `Individual` (合法的子代)。
    *   **作用**: 尝试生成子代，若生成的子代不满足约束（如参数量越界），则自动重试或重采样，确保返回可用个体。

### `src/utils/logger.py`
**作用**: 日志记录工具。

*   **`Logger` 类**: 封装了 Python `logging` 模块，提供标准化的日志输出格式（控制台+文件）。
*   **`OperationLogger` 类**: 专门用于将进化操作（交叉、变异详细信息）记录为 JSONL 文件。

---

## 5. Applications (应用脚本)

### `src/apply/predict.py`
**作用**: 使用训练好的模型进行单张图片预测。
*   **主要函数**: `predict_image(model_path, image_path, ...)`
    *   **输入**: 模型路径、图片路径、设备。
    *   **输出**: 无 (直接打印预测类别和置信度)。

### `src/apply/retrain_model.py`
**作用**: 对特定架构进行多次重训练以获取统计性能。
*   **主要函数**: `retrain_model(...)`
    *   **输入**: 模型路径、训练轮数、重复次数、数据集名称等。
    *   **输出**: `dict` (包含多次运行的平均准确率、标准差等统计信息)。

### `src/apply/compare_three_algorithms.py`
**作用**: 对比三种搜索策略（三阶段EA、传统EA、随机搜索）并绘制帕累托前沿。
*   **主要类**: `ThreeStageEA`, `TraditionalEA`, `RandomSearchAlgorithm`。
*   **输入**: 各自的运行参数（评估次数、种群大小等）。
*   **输出**: 生成包含性能对比的 JSON 结果和 PNG/PDF 图表。
