# NTK 计算逻辑调研总结

# 1.计算代码

``` python
import numpy as np
import torch


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds

```

## 2. 计算原理
该函数通过以下步骤计算 NTK 的条件数：
1.  **梯度提取**: 对每个输入样本调用 `logit.backward()`，获取模型权重相对于输出的梯度向量。
2.  **矩阵构造**: 计算梯度向量的内积（$JJ^T$），生成 NTK 矩阵。
3.  **指标计算**: 计算 NTK 矩阵的最大特征值与最小特征值的比值（条件数），用于衡量模型的可训练性。

## 3. 调用情况汇总
在 `prune_tenas.py` 中共有两处调用，分别用于评估算子剪枝前后的 NTK 变化。

### 调用位置
1.  `prune_tenas.py:111` (在 `prune_func_rank` 函数中)
2.  `prune_tenas.py:237` (在 `prune_func_rank_group` 函数中)

### 实际传入参数 (实参)
*   **xloader**: `loader` (即 `train_loader`)
*   **networks**: `[network_origin, network]` (包含原始网络和待评估的剪枝网络)
*   **recalbn**: `0` (不进行 BN 重校准)
*   **train_mode**: `True` (设置为训练模式计算梯度)
*   **num_batch**: `1` (仅使用 1 个 batch 以加速搜索)

## 4. 结论
NTK 的条件数在本项目中被用作架构搜索的无梯度指标（Zero-shot Proxy），通过观察剪枝掉某个算子后 NTK 条件数的变化，来评估该算子的重要性。

基于提供的论文《Neural architecture search on ImageNet in four GPU hours: A theoretically inspired perspective》，TE-NAS 框架引入了 **NTK（Neural Tangent Kernel，神经正切核）条件数**  作为衡量神经网络 **可训练性（Trainability）** 的指标。

以下是对 NTK 条件数计算过程、定义、物理含义以及公式推导的详细解析。

---

### 1. 核心定义与含义

**定义**：
NTK 条件数  定义为 NTK 矩阵的最大特征值  与最小（非零）特征值  的比值 ：


**含义**：

* **物理意义**：它反映了神经网络在梯度下降训练初期的优化难度。
* **指标解读**：
* ** 越小（越好）**：网络越容易训练，收敛速度越快，梯度流越健康。
* 
** 越大（越差）**：网络处于“病态（ill-conditioned）”状态，收敛极慢，甚至无法训练 。




* 
**在 NAS 中的作用**：论文发现  与测试集准确率呈现强烈的**负相关**（Kendall-tau = -0.42）。因此，TE-NAS 通过最小化该值来寻找高性能架构。



---

### 2. 计算过程详解 (Step-by-Step)

根据论文附录 A  和正文 3.1.1 节，计算一个特定网络架构  的 NTK 条件数的具体步骤如下：

1. **初始化 (Initialization)**：
* 构建神经网络架构 。
* 使用 **Kaiming Norm Initialization** 对网络参数  进行随机初始化 。


* **注意**：整个过程不涉及任何训练，权重是固定的。


2. **数据采样 (Data Sampling)**：
* 从训练集中随机采样一个 **Mini-batch** 的输入数据 ，Batch size 设置为 32 。




3. **计算雅可比矩阵 (Jacobian Calculation)**：
* 对于输入 ，计算网络输出  相对于所有参数  的雅可比矩阵 。
* 公式为： 。


* 这里  是最后一层的第  个神经元的输出， 是网络的一个参数。


4. **构建 NTK 矩阵 (NTK Matrix Construction)**：
* 计算 Gram 矩阵，即 NTK 矩阵 。
* 对于 Batch 中的任意两个样本点  和 ，NTK 定义为：



。


* 这是一个对称矩阵，描述了不同样本在参数空间中的梯度相关性。


5. **特征值分解 (Eigenvalue Decomposition)**：
* 对矩阵  进行特征值分解。
* 获取特征值并按降序排列： 。




6. **计算条件数 (Compute Condition Number)**：
* 提取最大特征值  和最小特征值 。
* 计算比值 。为了结果的稳定性，通常会重复多次（论文中为3次）取平均 。





---

### 3. 公式推导与理论支撑

论文之所以选择 NTK 条件数作为指标，是基于深度学习理论中关于**无限宽网络训练动力学**的研究（Lee et al., 2019）。推导逻辑如下：

#### 第一步：训练动力学的线性化

在无限宽度的极限下，神经网络在梯度下降训练过程中演化为一个线性模型。其输出 （在时间步 ）随训练的变化由以下微分方程控制 ：


* ：学习率。
* ：训练数据的 NTK 矩阵。
* ：训练标签。
* 当  时，预测值  将完全拟合标签  。



#### 第二步：特征值分解视角

为了分析收敛速度，我们将上述方程分解到  的特征空间中。对于第  个特征向量方向，收敛过程可以写为 ：


* 这里  是误差项。
* 显然，特征值  越大，误差项衰减越快（收敛快）；特征值越小，衰减越慢。

#### 第三步：学习率的限制与最慢收敛模式

理论研究（Lee et al., 2019）假设最大可行的学习率  受限于最大特征值 ，大致为 ：



如果学习率超过此值，训练将发散。

#### 第四步：推导条件数

将最大可行学习率  代入到对应最小特征值 （最难收敛的方向）的误差项中：


由此可见，整体网络的收敛速度取决于  的收敛速度，而这被比值  所控制 。

**结论**：
若  很大，意味着  很大而  很小。为了不让  方向发散，必须使用很小的学习率，这导致  方向（以及整个网络）收敛极慢。因此， 直接量化了网络的可训练性。

---

### 总结

| 步骤 | 公式/操作 | 目的 |
| --- | --- | --- |
| **1** |  | 获取参数空间对输出的敏感度（雅可比）。 |
| **2** |  | 构建 NTK 矩阵，描述样本间的梯度相关性。 |
| **3** |  | 计算特征值跨度。 |
| **推导核心** | 收敛速率  | 证明该指标越小，理论训练收敛越快。 |

TE-NAS 利用这一无需训练的指标，极大地降低了 NAS 的搜索成本，因为计算一次  仅需一次前向和反向传播（计算雅可比），而不需要完整的训练循环。

