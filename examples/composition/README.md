# 可组合重建：直接调用与可运行起点

从算法库根目录运行。Python >=3.10，复用现有 NumPy、SciPy、PyTorch、h5py、
PyYAML、matplotlib 环境；不需要 Agent SDK、模型服务或下载新数据集。

## 三个公开入口示例

```powershell
$compositionPython = 'D:/anaconda3/envs/inverse-agent-round1/python.exe'
$env:PATH = 'D:/anaconda3/envs/inverse-agent-round1/bin;D:/anaconda3/envs/inverse-agent-round1/Library/bin;D:/anaconda3/envs/inverse-agent-round1;' + $env:PATH
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $compositionPython -m inv_framework.runner --config examples/composition/ct_tv_haar.yaml --output outputs/composition_examples/ct
& $compositionPython -m inv_framework.runner --config examples/composition/mri_tv_haar.yaml --output outputs/composition_examples/mri
& $compositionPython -m inv_framework.runner --config examples/composition/deconvolution_tv_haar.yaml --output outputs/composition_examples/deconvolution
```

以上解释器为本机已安装环境；其他机器使用满足依赖的已激活 Python 环境。

也可直接使用 Python：

```python
import torch
from inv_framework.runner import run_experiment
torch.set_num_threads(1)
result = run_experiment("examples/composition/mri_tv_haar.yaml",
                        "outputs/composition_examples/mri_python")
print(result.status, result.converged, result.stopping_reason, result.iterations)
```

runner 调用 `run_ct_task(CTTaskRequest(...))`、`run_mri_task(MRITaskRequest(...))`
或 `run_deconvolution_task(DeconvolutionTaskRequest(...))`。这是算法库直接调用流程；
Agent 集成属于独立验收范围。

路径相对于 YAML 文件解析。`evaluation.reference` 仅在重建返回后用于指标，
不会传入求解、初始化或线圈估计。结果包含 request 对应的有效参数、停止原因、
轨迹、内层诊断及 reconstruction.npy。新运行使用独立输出目录。

## 数据来源与精度

三个示例均附带一个 seed=0 的匹配观测及参考文件：

| 示例数据目录 | 来源与实际设置 |
| --- | --- |
| `data/ct_seeded_dev` | 仓库随机椭圆函数，独立生成种子 2026110801，六个椭圆、64×64、固定乘 0.02；32 角 Torch 投影，Gaussian sigma=1% RMS(Ax)，float64 |
| `data/mri_standard_dev_seed0` | 既有 standard_mr_dev 对象、maps、mask；(1,4,128,128) 原生复数 Cartesian SENSE，centered ortho FFT；proper complex Gaussian sigma=0.002，complex128；ACS [48,80) 来自同一观测 |
| `data/camera_dev_seed0` | 既有 public_camera_levin 对象与 19×19 Levin 核1，原点 [9,9]；256×256 周期卷积，Gaussian sigma=0.005，float64 |

每个目录的 source.json 保存来源、噪声种子和预处理。显示与评价 data_range
固定为 CT 0.02、MRI/反卷积 1；MRI 默认评价幅值，重建数组保持复数。

数据已存在时无需准备。需要补建缺失的示例数据时，可执行：

```powershell
& $compositionPython examples/composition/prepare_data.py
```

该脚本只准备上述三个小型开发观测，已有完整目录直接复用，不覆盖半成品目录。
原 `data/ct_train0` 仍保留，其 train.npy[0] 对象种子缺失，已不作为主示例。
CT catalog 四个椭圆案例及 cone_3d__spheres_astra_12 保持原样；四个二维观测
对应同一 test[0]，不是四个独立对象。MR 模体和噪声重复也不代表独立受试者。

## 经验证可运行的数值起点

三个 YAML 都使用 ADMM + TV + Haar、L2、零初始化，TV/Haar 权重各 0.01，
两层 Haar、不惩罚低频，各向同性 TV、单位间距、末两空间轴独立通道定义。
外层上限 2000，tolerance=1e-5；固定 rho、rho_data、rho_regularizer、rho_haar、
rho_constraint 全为 1。cg_tolerance=1e-6；没有自适应 rho。

| 领域 | 子问题及预算 | 已有同输入校准 |
| --- | --- | --- |
| CT | float64；`subproblem_solver: cholesky`，4096 维正规系统一次分解；每步仍认证原真实残差目标 | CPU/CUDA 均在 1427 次达到 split_residual |
| MRI | complex128；Hermitian CG，上限300 | CPU 在386次达到 split_residual |
| 反卷积 | float64；物理卷积及 TV 均 periodic，可同时对角化时精确 FFT | CPU 在551次达到 split_residual |

校准记录位于 `outputs/composition_quality_20261108/development/calibration/`。
示例附带数组与这些校准的 seed0 数组一致。它们是针对这些输入验证过的起点，
不是最优参数，也不保证其他权重、噪声或几何在2000次以内收敛。

CT 的外部 float64 输入保留双精度；固定双线性插值以稀疏矩阵及转置复用。
Cholesky 的显式矩阵存储随未知量数平方增长：4096 维 float64 单矩阵约 128 MiB，
分解因子、构造与工作缓冲还需额外内存。它适用于这里的小型固定系统；
大型任务可使用既有 CG，周期卷积可使用适用的 FFT 路径。三种路径保持
`eta0 * max(1, ||b1||) / k²` 的真实内层残差认证，不通过放宽容差认可失败。

Windows Conda 环境按 [依赖说明](../../DEPENDENCIES.md) 设置 DLL 路径。
单线程可使用 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1。
CT 的 CUDA 示例需要现有 CUDA 环境；MRI、反卷积的公开示例使用 CPU。

## 组合入口与参数建议

`params.regularizer` 的五类单项为 `tv`、`haar_l1`、`tikhonov`、`h1`、
`gradient_huber`；`tv_haar` 为共享 ADMM 的联合项，独立使用 strength 与
haar_strength。三域 ADMM、PDHG、FISTA 的单项适配及 ADMM 联合适配见
[组合说明](../../docs/composition.md)和 `tests/test_composition_adaptation.py`。
已保留各域各单项的实际执行记录于本轮 development/search 下；它们是
执行证据，不是新的最优参数声明。

| 定义 | 约定与合法条件 |
| --- | --- |
| TV | isotropic/anisotropic，显式差分边界；MRI 使用原生复数梯度模 |
| Haar-L1 | 层数须与形状匹配；这里 level=2、penalize_lowpass=false |
| Tikhonov | 0.5 sum abs(x)^2；非负权重 |
| H1 | 0.5 sum abs(Dx)^2，使用声明的梯度边界 |
| gradient_huber | r²/(2 delta) 或 r-delta/2；delta>0，这里0.01；当前采用 Neumann |
| TV+Haar | ADMM；各权重非负，所有实际参与的固定分块 rho 严格为正 |

`candidates.yaml` 保留未按元素数归一化的 L2 参考范围：每项 reference=0.01，
建议六点 reference×10**[-2,-1.2,-0.4,0.4,1.2,2]；联合 q=[0.1,1,10]、
a=[0.25,0.75]，TV=q*a*TV_ref，Haar=q*(1-a)*Haar_ref。它们是建议候选，
不是保证收敛区间；不同数据项、单位、分辨率与差分定义的权重不能直接类比。
默认 rho=1，可在固定目标上参考0.1、1、10及固定分块比例校准；精度、残差
与成本达标后再固定设置。算法合法参数与建议搜索域见
[参数说明](../../docs/parameter_search.md)及各域 `list_capabilities()`。

`requests.py expand` 只生成请求，不执行搜索；不需要搜索服务。共同二次目标的
七方法配置仍可用 `requests.py quadratic --domain mri --method pdhg --output ...`。
四种线性基线只接受其支持的二次目标，不能把 TV、Haar 或 Poisson 更新强加给它们。

## 非默认保真项与边界

本轮保留的公开入口记录还包括：

- CT：FISTA + 单 TV 的 transmission_poisson_nll 与对数域 WLS，使用同一计数、
  非负约束与零初始化；`development/models/ct/fista_api_fixed4000/*/calibration/`。
- MRI：非均匀方差原生复数观测上的 WLS/L2 语义检查；加权 FISTA + TV 有达标记录。
  未加权 FISTA 的内层预算不足、加权 ADMM 的数值预算缺口均保留；没有完整模型质量排名。
- 反卷积：专用 `tv_admm` 的 Poisson NLL 与明确的 counts-background 二次近似，
  曝光并入已知 PSF。两项均显式使用相同零初始化、nonnegative=true；这是单 TV，
  不代表 Poisson 联合正则。记录在 development/models/deconvolution 下。

观测生成、保真项、预处理与实际目标分别记录。不同观测域的原始目标值不作横向
质量比较。`check_definitions.py` 提供少量边界、间距、复数 TV 与同方差等价检查。

success 只表示返回有限结果；converged 与停止原因共同说明数值达标情况。
maximum_iterations_reached、inner_budget_exhausted、inner_returned_precision、
joint_prox_inner_max_iterations 和 joint_prox_returned_precision 均不能视为收敛。
这些示例证明直接调用与组合可行性，不提供联合正则优于单项的结论；非 Cartesian、
三维组合质量与 Agent 集成不属于这些示例的已完成范围。
