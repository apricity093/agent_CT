# Landweber

## 一、算法身份

- **名称**：Landweber。
- **一句话简介**：使用增广算子及伴随进行最小二乘梯度迭代。
- **所属任务**：CT、MRI、反卷积。
- **方法类型**：非学习类，无需训练权重。
- **适用范围**：自动步长依据加权增广算子；CT 有界旧入口采用投影梯度。
- **代码来源**：[apricity093/agent_CT](https://github.com/apricity093/agent_CT)，包版本`0.1.0`，使用随本说明发布的源码（[版本声明](../../pyproject.toml)）。主要实现：[inv_framework/solvers/quadratic.py](../../inv_framework/solvers/quadratic.py)。

### 领域入口与目标

| 领域／入口 | 适用范围与实际目标 |
| --- | --- |
| CT / `landweber` | Augmented quadratic gradient iteration; bounded legacy path is projected L2. Unconstrained augmented weighted quadratic gradient iteration; legacy box path is projected unweighted L2 gradient. |
| MRI / `landweber` | Native complex 2D Cartesian SENSE 0.5 sum Q \|Ax-y\|^2 + 0.5 strength \|\|Lx\|\|^2; unconstrained |
| 反卷积 / `landweber` | Real scalar 2D known PSF; periodic or zero convolution boundary 0.5 sum Q \|Ax-y\|^2 + 0.5 strength \|\|Lx\|\|^2; unconstrained |

共享入口支持二维平行束CT、二维Cartesian SENSE及已知PSF二维反卷积。专用入口的维度、边界和正则约定保留各自语义；MRI使用原生复数和Hermitian伴随。共同线性基线只接收二次目标。

## 二、输入输出

### 最小示例数据

脚本生成固定4×4复数对象；单线圈maps、mask为1，中心正交FFT得到 kspace (1,1,4,4)。恒等正则权重0.08，无裁剪或强度拟合。

示例配置或生成脚本：[examples/quadratic_quickstart.py](../../examples/quadratic_quickstart.py)。

**正常输出**：(1,4,4) complex128 reconstruction.npy 和 result.json。

NPY用`numpy.load`读取；tensor bundle用`torch.load(..., weights_only=True)`读取。结果同时记录执行状态、实际迭代和停止原因；预算结束与数值收敛分别解释。

### 其他领域入口

- CT：[ct输入契约](../ct/data_format.md)，说明数组轴、几何/采样、单位及预处理。
- MRI：[mri输入契约](../mri/data_format.md)，说明数组轴、几何/采样、单位及预处理。
- 反卷积：[deconvolution输入契约](../deconvolution/data_format.md)，说明数组轴、几何/采样、单位及预处理。

## 三、运行方式

从算法库根目录执行。Python ≥3.10；基础依赖为PyTorch、NumPy、SciPy、h5py、PyYAML、matplotlib：

```bash
python -m pip install -e ".[dev]"
```

本例使用CPU。完整安装及Windows Conda设置见[依赖说明](../../DEPENDENCIES.md)。无需Agent SDK或大模型服务。

### 最小示例命令

```bash
python examples/quadratic_quickstart.py --domain mri --method landweber --output outputs/algorithm_examples/landweber
```

每次运行选择新的输出目录。该命令中的配置可能覆盖部分默认值；实际值以结果中的effective_params为准。

脚本的`--domain`可选ct、mri、deconvolution。线性CT示例读取64×64椭圆catalog，组合CT示例生成2×2匹配投影；MRI保留非平凡相位，反卷积使用已知小PSF。完整数据起点见[组合示例](../../examples/composition/README.md)。

## 四、参数

下面按入口列出参数，保留领域默认值及字段名差异。默认值由本仓库配置维护，属于工程运行起点；表中来源链接给出具体配置位置。显式参数覆盖实验配置，实验配置覆盖算法默认值。

建议范围不是硬边界；先固定观测尺度、离散定义和精度，再比较权重。步长、rho、内层预算以实际数值条件选择。

### CT：`landweber`

默认值来源：[本仓库配置](../../configs/algorithms/landweber.yaml)。未在YAML显式填写的值由[CT注册表](../../inv_framework/ct/registry.py)补齐。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Maximum outer iterations.；iterations | 50 | `configs/algorithms/landweber.yaml` | int；≥1 | [25, 50, 100] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| data_fidelity | Optimization data term; public sigma, weights or counts required by selected term；enum | l2 | `configs/algorithms/landweber.yaml` | str；l2 / gaussian_nll / wls | ['l2', 'gaussian_nll', 'wls'] |
| weight_source | Origin of public WLS weights; provided, known variance or matched post-log count approximation；enum | None | `configs/algorithms/landweber.yaml` | str；可为null；known_variance / provided / post_log_poisson / post_log_poisson_gaussian | ['known_variance', 'provided', 'post_log_poisson', 'post_log_poisson_gaussian'] |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | none | `configs/algorithms/landweber.yaml` | str；none / tikhonov / h1 | ['none', 'tikhonov', 'h1'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.0 | `configs/algorithms/landweber.yaml` | float；≥0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/algorithms/landweber.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/algorithms/landweber.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| step_size | Gradient step; automatic norm estimate when absent, coupled to actual curvature；inverse curvature | None | `configs/algorithms/landweber.yaml` | float；可为null；>0.0；0 < alpha * augmented_squared_norm_bound < 2 | [None] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Only registered updates; WLS requires public weights, Gaussian NLL public sigma, transmission NLL public counts/I0 and nonnegative constraints.

### MRI：`landweber`

默认值来源：[本仓库配置](../../configs/mri/algorithms/landweber.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Iteration budget; actual stop may be earlier；iterations | 200 | `configs/mri/algorithms/landweber.yaml` | int；≥1 | [100, 200, 400] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/mri/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/mri/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| step_size | Gradient step; automatic norm estimate when absent, coupled to actual curvature；inverse curvature | None | `configs/mri/algorithms/landweber.yaml` | float；可为null；>0.0 | [None] |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | none | `configs/mri/algorithms/landweber.yaml` | str；none / tikhonov / h1 | ['none', 'tikhonov', 'h1'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.0 | `configs/mri/algorithms/landweber.yaml` | float；≥0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/mri/algorithms/landweber.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/mri/algorithms/landweber.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Quadratic 0.5 sum W |Ax-y|^2; real inverse-variance weights, identity if absent. WLS requires explicit weights. Shared MRI: 2D Cartesian SENSE; deconvolution: known PSF, gain=1, background=0.

### 反卷积：`landweber`

默认值来源：[本仓库配置](../../configs/deconvolution/algorithms/landweber.yaml)。

| 参数名 | 作用与单位 | 默认值 | 默认值来源 | 允许值或必要约束 | 推荐范围 |
| --- | --- | --- | --- | --- | --- |
| num_iterations | Iteration budget; actual stop may be earlier；iterations | 200 | `configs/deconvolution/algorithms/landweber.yaml` | int；≥1 | [100, 200, 400] |
| min_value | Optional lower image constraint, coupled to upper bound；image intensity | None | `configs/deconvolution/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| max_value | Optional upper image constraint, coupled to lower bound；image intensity | None | `configs/deconvolution/algorithms/landweber.yaml` | float；可为null | 固定数值/采集条件，按任务显式覆盖 |
| step_size | Gradient step; automatic norm estimate when absent, coupled to actual curvature；inverse curvature | None | `configs/deconvolution/algorithms/landweber.yaml` | float；可为null；>0.0 | [None] |
| regularizer | Prior selector; ADMM tv_haar is a joint objective with separate weights；enum | none | `configs/deconvolution/algorithms/landweber.yaml` | str；none / tikhonov / h1 | ['none', 'tikhonov', 'h1'] |
| strength | Regularizer weight on the stated unnormalized sum；objective/regularizer | 0.0 | `configs/deconvolution/algorithms/landweber.yaml` | float；≥0.0 | [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] |
| tolerance | Relative numerical stopping accuracy；relative residual | 1e-06 | `configs/deconvolution/algorithms/landweber.yaml` | float；≥0.0 | 固定数值/采集条件，按任务显式覆盖 |
| boundary | 正则差分边界；物理卷积边界由model单独声明；dimensionless | neumann | `configs/deconvolution/algorithms/landweber.yaml` | str；neumann / periodic | ['neumann', 'periodic'] |

保真项模式：`selectable`；可执行项：`['l2', 'gaussian_nll', 'wls']`。Quadratic 0.5 sum W |Ax-y|^2; real inverse-variance weights, identity if absent. WLS requires explicit weights. Shared MRI: 2D Cartesian SENSE; deconvolution: known PSF, gain=1, background=0.

### 耦合条件

权重对应未按像素数平均的目标。Haar层数须适配尺寸；rho为正且与正则权重独立；步长和CG精度依据实际算子与系统。MRI复数变量不接受实数盒约束，具体组合按入口条件执行。

目标与数值操作见[共享数学说明](../shared_mathematics.md)，任务相关选参见[参数说明](../parameter_search.md)。
