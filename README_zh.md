# 基于 Aging Evolution 和 NTK 评估的神经架构搜索

本项目实现了一套演化式 NAS 框架：先用 NTK 条件数对候选架构进行零成本评估，再通过分阶段训练选出高性能模型。

## 特性
- **老化进化 (Aging Evolution)** 搜索，包含交叉/变异和重复过滤。
- **NTK 零成本代理**：在参数约束下快速打分候选架构。
- **两阶段训练**：先短训筛选，再对少量最佳模型全量训练。
- **可扩展搜索空间**：可变长编码，支持通道/分组/池化/SENet/跳连/卷积核/扩张等选项。
- **实验工具**：随机搜索对比、相关性实验、NTK 曲线绘图等。
- **轻量化测试**：`tests/` 提供重依赖 stub，便于 CI。

## 目录结构
- `src/main.py` — CLI 入口，启动 NAS 搜索/训练。
- `src/configuration/config.py` — 全局超参与设置。
- `src/core/` — 编码工具与搜索空间采样。
- `src/search/` — 进化逻辑（老化进化、变异、交叉）。
- `src/engine/` — NTK 评估、最终训练与训练器。
- `src/models/` — 搜索架构的网络构建。
- `src/utils/` — 日志、约束、个体生成辅助。
- `src/apply/` — 实验与绘图脚本。
- `tests/` — Pytest 套件，含重依赖 stub。

## 安装
```bash
python -m venv .venv
.venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> 注意：torch/torchvision/numpy 需与你的 Python 版本和 GPU 环境兼容；若无 GPU 会自动回退 CPU。

## 快速开始
在 CIFAR-10 上运行 NAS 搜索：
```bash
python src/main.py --dataset cifar10
```
常用可选参数（见 `src/main.py` 的 `parse_args`）：
- `--optimizer {adamw,sgd}`
- `--dataset {cifar10,cifar100,imagenet}`
- `--imagenet_root PATH`（使用 ImageNet 时必需）
- `--seed INT`
- `--resume CHECKPOINT_PATH`（恢复搜索）
- `--no_final_eval`（如集成跳过最终训练）

## 核心配置要点（`configuration.config.Config`）
- 搜索：种群大小、最大代数、锦标赛规模、交叉/变异概率。
- 搜索空间：unit/block 数范围，通道/分组/池化/SENet/激活/跳连/卷积核/扩张等选项。
- 约束：参数量上下限及按数据集的覆盖。
- 训练：batch、优化器默认值、warmup+cosine 调度、早停。
- IO：日志/检查点/TensorBoard 目录，失败个体保存等。

## 工作流
1) **初始化种群**：`core.search_space.population_initializer` 在约束内采样有效编码。
2) **NTK 打分**：`engine.evaluator.NTKEvaluator` 构网、检查参数量、计算 NTK 条件数（越低越好），非法/超限模型直接惩罚。
3) **进化循环**：`search.evolution.AgingEvolutionNAS` 锦标赛选父、`utils.generation.generate_valid_child` 交叉/变异生成子代，评估 NTK，维护 FIFO 种群。
4) **筛选训练**：Top NTK 先短训，Top 若干再全量训练（`engine.evaluator.FinalEvaluator` + `engine.trainer.NetworkTrainer`）。
5) **产物**：检查点、NTK 历史 JSON/图、训练模型、日志等按配置目录输出。

## 实验/绘图脚本
位于 `src/apply/`，可直接运行，例如：
```bash
python src/apply/compare_evolution_vs_random.py --max_eval 50 --pop_size 10
python src/apply/plot_ntk_curve.py --input logs/ntk_history.json
python src/apply/ntk_correlation_experiment.py
```
脚本默认使用 Agg 后端，可在无图形界面环境运行。

## 测试
```bash
PYTHONPATH=src pytest
```
测试使用 `tests/conftest.py` 中的 stub（numpy/torchvision/pandas/scipy/PIL）。新增依赖时保持可选或提供 stub，避免破坏 CI。

## 扩展搜索空间
1) 在 `Config` 中新增选项/常量。
2) 更新 `core.encoding.BlockParams` / `BLOCK_PARAM_COUNT` 和 `core.search_space` 的采样。
3) 在 `core.encoding.Encoder.validate_encoding` 加入校验。
4) 在 `models.network` 中支持新参数的建模。

## 设备与内存
- 所有 CUDA 操作前先 `torch.cuda.is_available()`，支持 CPU 回退。
- 大计算后可调用 `engine.evaluator.clear_gpu_memory()` 清理显存。
- 参数量需落在配置范围，避免 OOM 或无效编码。

## 日志与输出
- 日志：`logs/`（`utils.logger` 生成时间戳文件）。
- TensorBoard：`runs/`（如启用）。
- 检查点：`checkpoints/`（搜索/训练模型）。
- NTK 历史：`logs/ntk_history.json` 与曲线 PNG。

## 健壮性提示
- 对缺失数据集/检查点要清晰报错（ImageNet 需提前准备目录）。
- 默认/测试路径不应依赖网络下载或 GPU。
- CLI 可选参数应有合理默认，不应因缺失字段崩溃。
