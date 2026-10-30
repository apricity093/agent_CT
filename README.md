# inv_framework

`inv_framework` 是一个面向 X-ray CT 重建与评测的轻量 Python 框架。它将 CT
前向/伴随算子、经典重建算法、版本化测试数据、参数验证、收敛诊断和公平比较协议
组织为统一接口，并提供 `invct` 命令行工具。

当前源码提供 17 个 ordinary-CT solver、纯 PyTorch 2D parallel-beam Radon、ASTRA
3D/FDK 后端，以及可供上层 inverse-problem Agent 程序化读取的算法 metadata。

## CT 重建模型

离散线性 CT 前向模型为：

```text
y = A x + n
```

- `x`：二维衰减图或三维体数据；
- `A`：由扫描几何确定的正投影算子；
- `y`：探测器测量；
- `n`：高斯噪声、Poisson 光子噪声或其他误差。

常见的正则化重建目标为：

```text
min_x  0.5 * ||A x - y||_2^2 + lambda * R(x)
```

算法比较必须保持投影数据、几何、预处理、分辨率、初始化、数值精度和预算协议一致。

## 支持的算法

| CLI 名称 | 算法 | 几何 | 主要配置 | 后端 |
| --- | --- | --- | --- | --- |
| `fbp` | Filtered Backprojection | 2D parallel | `scale` | PyTorch CPU/CUDA |
| `sirt` | SIRT | 2D parallel | `num_iterations`、上下界 | PyTorch CPU/CUDA |
| `landweber` | Landweber | 2D parallel | `num_iterations`、`step_size` | PyTorch CPU/CUDA |
| `cgls` | CGLS | 2D parallel | `num_iterations`、`tol` | PyTorch CPU/CUDA |
| `lsqr` | LSQR | 2D parallel | `num_iterations`、`atol`、`btol` | PyTorch CPU/CUDA |
| `sart` | SART | 2D parallel | `block_size`、`relaxation` | PyTorch CPU/CUDA |
| `os_sart` | Ordered-subsets SART | 2D parallel | `subset_count`、`relaxation` | PyTorch CPU/CUDA |
| `mlem` | MLEM | 2D parallel | `num_iterations`、`initial_value` | PyTorch CPU/CUDA |
| `osem` | OSEM | 2D parallel | `subset_count`、`initial_value` | PyTorch CPU/CUDA |
| `tikhonov` | Tikhonov CG | 2D parallel | `reg_strength`、`tolerance` | PyTorch CPU/CUDA |
| `tv_fista` | TV-FISTA | 2D parallel | `reg_strength`、TV proximal 参数 | PyTorch CPU/CUDA |
| `fista` | 通用 FISTA | 2D parallel | `regularizer`、`strength`、回溯 | PyTorch CPU/CUDA |
| `admm` | ADMM | 2D parallel | `regularizer`、`strength`、`rho`、CG | PyTorch CPU/CUDA |
| `pdhg` | 惩罚 PDHG | 2D parallel | `regularizer`、`strength` | PyTorch CPU/CUDA |
| `constrained_pdhg` | 残差球约束 TV PDHG | 2D parallel | `epsilon`、`epsilon_source`、TV | PyTorch CPU/CUDA |
| `asd_pocs_tv` | SART2D 约束 TV 启发式 | 2D parallel | `epsilon`、TV、relaxation | PyTorch CPU/CUDA |
| `fdk` | Feldkamp-Davis-Kress | 3D circular cone | filter、short scan、supersampling | ASTRA CUDA |

查询 registry、默认参数和能力约束：

```bash
invct list-solvers
invct list-solvers --json
invct list-regularizers --json
```

canonical solver ID 固定为：

```text
fbp, sirt, landweber, cgls, lsqr, sart, os_sart, mlem, osem,
tikhonov, tv_fista, fista, admm, pdhg, constrained_pdhg, asd_pocs_tv, fdk
```

`tikhonov` 可选恒等 Tikhonov 或 H1，`tv_fista` 固定搭配 TV 正则。MLEM/OSEM 只接受
显式的非负 emission/count observation model；不能把 X-ray `line_integral` 或
`log_projection` 当成计数数据。FDK 要求 `cone_3d`、cubic volume、PyTorch CUDA 和
ASTRA CUDA。

## 安装

要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/apricity093/agent_CT.git
cd agent_CT
python -m pip install -e ".[dev]"
```

安装后可使用控制台入口，也可使用等价的模块入口：

```bash
invct --help
python -m inv_framework --help
invct run --help
invct data --help
```

Windows Conda 环境在调用下列命令前应激活环境，并设置其 DLL 路径和线程数：

```powershell
$env:PATH = "$env:CONDA_PREFIX/Library/bin;$env:CONDA_PREFIX;" + $env:PATH
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
```

本文多行 Bash 示例使用 `\` 续行；PowerShell 请合并为单行或使用反引号续行。

FDK 还需要安装带 CUDA 支持的
[ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox)。二维算法可在 CPU 或
CUDA 上运行；FDK 不会回退到语义不同的 CPU 实现。

## 数据查询与校验

默认数据根目录为 `test/data`。查询 catalog：

```bash
invct data list
invct data list --dimension 2
invct data list --geometry parallel_2d --tag quality
invct data show parallel_2d/tissue_breast_dense_clean_128
```

校验 manifest、HDF5 shape 和 SHA256：

```bash
invct data validate
invct data validate parallel_2d/tissue_breast_dense_clean_128
```

外部 catalog 可通过 `--root PATH`（数据子命令）或 `--data-root PATH`（`run`）传入。
路径指向包含 `catalog.json` 的目录，例如：

```bash
invct data list --root /path/to/ct-data --json
invct data show parallel_2d/disk_analytic_32 --root /path/to/ct-data
```

每个 case 的 HDF5 使用以下统一键：

| 路径 | 含义 |
| --- | --- |
| `truth/x` | ground truth，shape 为 `(B, *domain_shape)` |
| `measurement/y_clean` | 无噪测量 |
| `measurement/y_observed` | solver 的实际输入 |
| `masks/roi` | 可选图像 ROI |
| `masks/valid_measurement` | 可选有效测量 mask |

## 运行单个算法

每个 solver 使用自己的严格 YAML 配置。未知字段、错误类型、无效参数、几何或观测域
不兼容会在数值迭代前失败。

CPU 示例：

```bash
invct run cgls \
  --case parallel_2d/tissue_breast_dense_clean_128 \
  --config configs/algorithms/cgls.yaml \
  --out artifacts/cgls_dense_cpu \
  --device cpu
```

CUDA 示例：

```bash
invct run tv_fista \
  --case parallel_2d/tissue_breast_sparse_poisson_128 \
  --config configs/algorithms/tv_fista.yaml \
  --out artifacts/tv_fista_sparse_cuda \
  --device cuda
```

FDK 示例：

```bash
invct run fdk \
  --case cone_3d/spheres_astra_12 \
  --config configs/algorithms/fdk.yaml \
  --out artifacts/fdk_cone_cuda \
  --device cuda
```

MLEM/OSEM 必须使用明确声明 `nonnegative_counts` 或 `intensity` 的 emission catalog。
不要将上面的 transmission case 替换进统计重建命令。SART/OS-SART 的
`block_size`/`subset_count` 还必须与公开 view 数形成合法 partition。

输出目录非空时命令默认拒绝运行，防止新旧 manifest 混合。只有确认替换该次 run
产物时才显式添加 `--overwrite`。

### 选择 solver 与正则项

`invct run` 的位置参数选择 solver，YAML 的 `parameters.regularizer` 选择受支持的
正则项；当前 CLI 没有 `--regularizer` 或 `--operator` 参数。已有组合示例：

```bash
invct run fista --case parallel_2d/disk_analytic_32 --config configs/examples/fista_haar_l1.yaml --out artifacts/fista_haar_cpu --device cpu
invct run tikhonov --case parallel_2d/disk_analytic_32 --config configs/examples/tikhonov_h1.yaml --out artifacts/tikhonov_h1_cpu --device cpu
```

直接重建不要求迭代 solver，无正则算法不要求 regularizer。CLI/runtime 的 CT operator
由实际 case geometry 构建；其他 Python operator 类的存在不代表它们已经接入 CLI。
几何、投影数和观测噪声的变化需要相应输入数据，不能只修改算法参数或描述字段。

### 运行预算与停止策略

```bash
invct run fista --case parallel_2d/disk_analytic_32 --config configs/algorithms/fista.yaml --out artifacts/fista_budget_cpu --device cpu --max-iterations 50 --max-forward-calls 1500 --max-adjoint-calls 1500 --max-wall-seconds 60
```

| 选项 | 含义 |
| --- | --- |
| `--max-iterations N` | 外层迭代预算；算法迭代数仍由配置指定并受预算限制 |
| `--max-forward-calls N` / `--max-adjoint-calls N` | A/A* 调用上限，包含初始化、参数估计和终点工作 |
| `--max-wall-seconds S` | 协作式 runtime 时间预算；单次算子调用不能被抢占 |
| `--stopping-policy PATH` | schema `1.0`、mode `solver_native_and_endpoint` 的 JSON 停止策略 |
| `--parameter-sources PATH` | 参数名到来源标签的 JSON mapping |
| `--observation-domain DOMAIN` | 选择 case 中已提供的观测域，不生成新的观测数据 |
| `--fixed-compute` | 关闭 solver 提前停止，用于声明了固定计算预算的实验 |

`--fixed-compute` 不取消预算或数值异常检查。`--max-wall-seconds` 不等于进程硬超时；
硬超时由上层 Agent 的 `budget.timeout_s` 控制。达到迭代上限或预算耗尽不表示收敛。
CLI 返回码为：成功执行 `0`，无效参数/配置 `2`，运行失败或资源中断等 `1`；
优化收敛状态需要继续读取 diagnostics。

## 离线评估

`eval` 从已保存的 tensor bundle 重算指标，不重新运行 solver，也不依赖原机器上的
projector 或 ASTRA：

```bash
invct eval \
  --run artifacts/cgls_dense_cpu \
  --protocol configs/protocols/traditional_quality.yaml
```

评估 protocol 使用显式 `{min: ...}` 或 `{max: ...}` 表示阈值方向。存在私有 ground
truth 时可报告 RMSE、PSNR、SSIM 和 relative error；Agent public staging 不向 backend
暴露 truth，此时 backend 只报告数据一致性、优化和资源指标，图像质量由独立 evaluator
计算。

## 运行 benchmark

版本化 suite 将算法配置列表与 case 列表分组，并按几何和观测模型过滤不兼容组合：

```bash
invct bench --suite configs/benchmarks/traditional_quality.yaml
```

公平比较协议可在不执行重建的情况下校验：

```bash
invct protocol-check \
  --protocol configs/fair_protocols/equal_operator_calls_v1.yaml
```

支持固定默认值、等 trial、等调参时间、等 operator calls、公共验证集，以及单独标记的
离线 oracle upper bound。在线选择不得使用测试 truth。结果按图像质量、数据一致性、
优化行为、计算效率和鲁棒性分别报告，不合成为无依据的总分。

## 在 inverse-problem Agent 中使用

本仓库可以独立运行，也可以作为
[`inverse_problem_agent`](https://github.com/zoe5xy/inverse_problem_agent) 的 CT backend。
在 Agent 项目根目录执行：

```bash
inverse-agent backend-info \
  --modality ct \
  --ct-repo external/ct-benchlab

inverse-agent list-cases \
  --modality ct \
  --ct-repo external/ct-benchlab

inverse-agent list-algorithms \
  --modality ct \
  --ct-repo external/ct-benchlab

inverse-agent list-ct-components \
  --ct-repo external/ct-benchlab

inverse-agent enumerate-ct-compositions \
  --request request.json \
  --ct-repo external/ct-benchlab

inverse-agent compose-ct-request \
  --request request.json \
  --composition algorithm.json \
  --ct-repo external/ct-benchlab

inverse-agent select-ct-candidates \
  --request request.json \
  --ct-repo external/ct-benchlab

inverse-agent check-capability \
  --request request.json \
  --ct-repo external/ct-benchlab

inverse-agent validate-run \
  --request request.json \
  --ct-repo external/ct-benchlab

inverse-agent run-experiment \
  --request request.json \
  --ct-repo external/ct-benchlab
```

组件命令由包含组合接口的主项目版本提供，安装本 CT 包不会安装 `inverse-agent`。
`enumerate-ct-compositions` 返回组合、参数建议、准入结果和候选请求；
`compose-ct-request` 输出完整 RunRequest JSON，以上两者均不执行重建。
保存组装结果后，后续检查和运行必须使用该结果文件。

`algorithm.json` 示例：

```json
{
  "algorithm_id": "fista",
  "components": {"regularizer": "tv"},
  "parameters": {"strength": 0.002, "num_iterations": 100}
}
```

省略组件字段继承实现；显式 `solver: null` 用于直接方法，
`regularizer: null` 要求无正则实现，省略或 null operator 使用 case operator。
三者无需同时指定。显式 operator 必须与实际 case 一致；每个选中请求应使用
独立 run_id 和空 output_dir。Python 接口、支持范围及版本要求见
[Agent 用法](docs/agent_usage.md)。

完整执行路径为：

```text
RunRequest
  -> Agent candidate selection and compatibility filtering
  -> Agent parameter/budget validation
  -> CTAdapter
  -> invct subprocess
  -> CT solver and diagnostics
  -> Agent convergence gate and independent evaluation
```

Agent 负责问题解释、候选选择、参数来源、尝试谱系和跨运行比较；CT backend 负责数值
合法性、solver-native 停止判据、operator-call 记账及低层诊断。达到最大迭代数或预算
耗尽不会被 Agent 改写为 `converged`。

## 输出与诊断

单次成功运行生成：

```text
run-directory/
├── reconstruction.pt
├── metrics.json
├── diagnostics.json
├── manifest.json
├── comparison.png
└── artifacts.sha256
```

`reconstruction.pt` 保存 CPU tensor bundle，便于在没有原运行 GPU 的机器上离线复评。
manifest 记录 Python、PyTorch、平台、CUDA、Git revision、dirty patch hash、输入来源、
参数和参数来源。失败运行写入结构化 failure 文件，不会从 benchmark 中静默删除。

常见 convergence status 包括：

```text
converged
completed_valid
max_iterations
stalled
diverged
resource_exhausted
numerical_error
invalid_parameters
unavailable
```

直接算法在输出有限、shape 正确且参数有效时报告 `completed_valid`，不是数学意义上的
`converged`。诊断同时记录 stopping reason、objective/residual、最终残差、
forward/adjoint calls、运行时间和资源信息。

预算中断统一为 `resource_exhausted`；算子次数与 runtime wall 的区别通过
`operator_call_budget_exhausted` / `wall_time_budget_exhausted` 停止原因表达。

L2 数据项下，Landweber 的固定步长条件为 `0 < step_size < 2 / ||A||^2`，
FISTA 的标准固定步长条件为 `0 < step_size <= 1 / ||A||^2`。省略步长时使用
确定性有限 power iteration 估计；该估计不是谱范数上界。当前 FISTA/TV-FISTA
对试探步长检查实际光滑项 majorization，失败时减半回溯。参数估计、回溯与
solver 迭代的 operator calls 分别记账；有限 TV proximal 的限制见组合算法文档。

## 可组合算法与 Agent

当前完整支持矩阵及数学、数据和参数要求见 [已支持的 CT 组件组合](docs/supported_combinations.md)。

CT 请求将 `solver`、`operator`、`regularizer`、`data_fidelity`、
`observation_model`、`constraints` 与公开 `noise_parameters` 分开表达。
Agent 先发现候选，再校验物理域、数据字段和数学兼容性，最后装配并执行。
已登记的二次组合可使用 L2、已知方差 Gaussian NLL 或固定 WLS；
FISTA/ADMM/PDHG 还提供纯透射 Poisson NLL 路径。正则包括无正则、恒等
Tikhonov、H1、两种 TV、梯度 Huber 与 Haar detail-L1。
这表示受兼容性约束的组件组合，并不表示任意 solver、prior 与几何均可互换；
执行完成、优化收敛和重建质量分别记录。下方用固定种子的真实图像展示部分路径。

FISTA/H1/Haar 的参数和兼容性见 [用法](docs/composable_reconstruction.md)，
ADMM/PDHG/Huber 见 [分裂算法](docs/splitting_solvers.md)，
Gaussian NLL、WLS 与透射 Poisson NLL 见 [保真项](docs/data_fidelity.md)。
约束 TV 的显式半径、来源、非负性及停止语义见 [库用法](docs/constrained_tv.md)，
ASD 变体及参考许可见 [来源](docs/constrained_tv_sources.md)。

主项目提供 17 项精确能力检查和真实 Agent 路径；安装、请求参数、runtime wall
与硬超时语义见 [Agent 用法](docs/agent_usage.md)。主项目通过 `pyproject.toml`
固定本库提交；固定来源安装检查与数值精度验收分别记录。P7/P8 未启动。

有限预算结果不构成最佳质量排名。独立 pilot 与冻结配置示例见
[质量示例](docs/quality_example.md)。

## 已知精度与成本限制

约束 PDHG 依赖凸 TV 与 A/D 谱上界，必须同时满足可行性和原生 fixed-point
条件才报告 converged。ASD 为 SART2D feasibility-backtracking 变体，无全局最优保证。
`tv_fista` 的 TV proximal 为有限 FGP；通用 `fista` 的 TV/受约束 Haar
使用带 gap 及误差目标的联合近端，内层精度不足不会标为收敛。

## 最新 Agent 端到端结果

以下选取少量真实 Agent 结果，展示 **solver 切换、正则切换、保真项切换及
3D 几何入口**。

### 2D Gaussian：切换 solver 与 regularizer

128×128 组织 phantom，48 views 覆盖 180°，匹配的
`ParallelBeamRadon2D` 生成观测并重建。Gaussian σ=0.002，使用已知方差 NLL、
非负约束及零初始化。FISTA/ADMM/PDHG 的恒等 Tikhonov 对照共用
λ=7499.99953125；FISTA 的 TV 与 Haar-L1 是另行记录参数的不同目标。
参数通过独立种子 `271828` 的测量域双折验证选择，truth 只用于生成与最终评价。

![固定种子Gaussian观测下的solver和正则组合重建](docs/assets/component-showcase/gaussian_components.png)

| Solver + regularizer | Data fidelity | λ | PSNR (dB) | SSIM | RMSE | 优化状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FISTA + 恒等 Tikhonov | Gaussian NLL | 7499.99953125 | 21.70 | 0.476 | 3.29036e-4 | resource_exhausted |
| ADMM + 恒等 Tikhonov | Gaussian NLL | 7499.99953125 | 25.07 | 0.340 | 2.23034e-4 | resource_exhausted |
| PDHG + 恒等 Tikhonov | Gaussian NLL | 7499.99953125 | 23.51 | 0.505 | 2.67076e-4 | resource_exhausted |
| FISTA + isotropic TV | Gaussian NLL | 0.0749999953125 | 22.91 | 0.497 | 2.86273e-4 | resource_exhausted |
| FISTA + Haar detail-L1，level=2 | Gaussian NLL | 14.9999990625 | 23.36 | 0.507 | 2.71626e-4 | resource_exhausted |


这些案例说明组合能够通过 Agent 链路执行并保存结果，不表示已经收敛或质量最优。

### 2D Poisson：同一 counts 切换 data fidelity

保持 128×128、48 views、同一非负 TV 和零初始化，使用同一份纯透射计数：
`counts ~ Poisson(1000 * exp(-Ax))`，s=1，无电子噪声或背景。
post-log WLS 采用固定 1 个计数的 log-floor；pre-log NLL 直接读取原始 counts。
两者 λ 均为 0.000261538623046875，原始 counts 不被修改。

![同一Poisson计数下PDHG与TV的两种保真项重建](docs/assets/component-showcase/poisson_fidelity.png)

| Solver + regularizer | Data fidelity | PSNR (dB) | SSIM | RMSE | 优化状态 |
| --- | --- | ---: | ---: | ---: | --- |
| PDHG + isotropic TV | post-log WLS，近似模型 | 2.70 | 0.122 | 2.93123e-3 | resource_exhausted |
| PDHG + isotropic TV | pre-log 透射 Poisson NLL | 2.64 | 0.124 | 2.95284e-3 | resource_exhausted |

该噪声条件下图像质量较差，两条路径都达到时间预算；这里展示保真项接入与输出保存，
**不将其列为高质量重建成功案例**。

### 3D cone-beam：FDK

直接读取历史 `cone_3d/tissue_volume_clean_64` 的同一份 HDF5 truth、原始投影和几何，
使用 `ASTRAFDKOperator3D + FDK`、无显式正则、Ram-Lak、
`short_scan=false`、`voxel_supersampling=1`，经真实 Agent 调用。
上排为 reference，下排为重建，依次显示中心 axial、coronal 和 sagittal 切面。

![原README同一3D观测的FDK重建及三个正交切面](docs/assets/component-showcase/cone_fdk.png)

| Operator + method | Regularizer | PSNR (dB) | SSIM | RMSE | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| ASTRA cone-beam + FDK | 无 | 30.68 | 0.818 | 1.16949e-4 | completed_valid |

独立切面图已保存：[axial](docs/assets/component-showcase/fdk_axial.png)、
[coronal](docs/assets/component-showcase/fdk_coronal.png)、
[sagittal](docs/assets/component-showcase/fdk_sagittal.png)。
服务器同时保留该次 `fdk/backend/reconstruction.pt`、输入 HDF5、请求、指标及诊断，
便于后续复用；本仓库包含图版、独立 PNG、PDF/SVG 和
[精确参数与成本](docs/assets/component-showcase/selected_results.json)。

### 展示范围与完整实验

- 所有面板统一灰度窗 `[0, 0.004]`，无裁剪、平滑或逐图归一化；显示窗外数值仅在显示时饱和。
  PSNR/SSIM 使用固定 data range=0.004、完整图像/体数据，RMSE 使用原始数值。
  3D SSIM 为轴向二维切片均值。

## Python API

CLI 与 Python API 使用相同的 case 和 operator：

```python
import torch

from inv_framework.benchmarks import load_ct_case
from inv_framework.operators.ct import ParallelBeamRadon2D
from inv_framework.solvers import CGLSSolver

device = "cuda" if torch.cuda.is_available() else "cpu"
case = load_ct_case(
    "parallel_2d/tissue_breast_dense_clean_128",
    device=device,
)
angles = torch.tensor(case.geometry["angles_rad"], device=device)
operator = ParallelBeamRadon2D(
    image_size=case.truth.shape[-1],
    angles=angles,
    device=device,
)
solver = CGLSSolver(num_iterations=25, min_value=0.0, max_value=0.02)
reconstruction = solver.solve(case.measurement, operator)
```

自定义线性 CT 算子需要实现 `LinearOperator.forward()` 与
`LinearOperator.adjoint()`；自定义 solver 应实现
`InverseProblemSolver.solve(y, operator, **kwargs)`。

## 项目结构

```text
agent_CT/
├── inv_framework/
│   ├── cli.py                 # invct 命令解析
│   ├── ct_runtime.py          # registry、run、eval、bench
│   ├── benchmarks/            # 版本化 case API
│   ├── operators/             # forward / adjoint / CT operators
│   ├── solvers/               # 经典、子集、统计与正则化算法
│   ├── regularizers/          # Tikhonov、TV、Haar、Huber
│   └── utils/                 # PSNR、SSIM
├── configs/
│   ├── algorithms/            # solver YAML
│   ├── benchmarks/            # benchmark suite
│   ├── fair_protocols/        # 公平比较协议
│   └── protocols/             # 评估阈值
├── test/                      # CT runtime 与集成测试
├── tests/                     # 算子和既有回归测试
├── examples/
├── tools/
└── pyproject.toml
```

## 开发与测试

Windows CUDA 环境需先导入 torch 再启动 pytest，并使用环境的 Library/bin PATH
和单线程设置。例如快速 registry 检查：

```powershell
python -c "import torch; import pytest; raise SystemExit(pytest.main(['test/test_invct_cli.py::test_help_and_registry_are_stable', 'test/test_registry_diagnostics.py::test_registry_build_config_bijection_is_json_stable', 'test/test_batch10_fair_protocols.py', '-q']))"
```

这组命令只检查命令发现、registry/config 对应及公平协议，不执行真实重建。
下列完整测试和快速数值测试则包含真实 solver 调用。

全部测试包含高成本独立参考与长程高精度 CT 专项，只有明确需要完整范围时运行：

```bash
python -m pytest -q test tests
```

运行 CLI、registry、参数和收敛相关的快速验证：

```bash
python -m pytest -q \
  test/test_invct_cli.py \
  test/test_registry_diagnostics.py \
  test/test_batch6_parameter_validation.py \
  test/test_convergence_protocol.py
```

验证公平协议和紧凑 benchmark 证据：

```bash
python -m pytest -q \
  test/test_batch10_fair_protocols.py \
  test/test_batch13_budgeted_benchmark.py
```

小规模代码验证与完整 CT benchmark 应分开执行。修改 forward/adjoint operator 时还应
执行数值伴随一致性检查：

```text
<A x, y> approximately equals <x, A* y>
```

### 构建可安装包

已有 setuptools（至少 68）和 wheel 的环境可直接构建本地 wheel，不解析或安装
运行时依赖：

```bash
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir artifacts/wheels
```

wheel 携带 Python 数值模块与 17 个 `configs/algorithms/*.yaml`。
README 中的 `configs/examples`、benchmark/protocol 文件、`test/data` 和 docs
按源码 checkout 使用，不随 wheel 提供；安装包运行时需要显式提供数据目录和配置。
主项目的组件发现、组合和请求组装代码还需在主项目中单独提交。
固定版本安装必须使用已推送且可获取的 commit，再同步主项目依赖引用，不能用
本地 dirty checkout 的结果代替固定来源安装验收。

## 能力边界与数据许可

- `ASTRAFDKOperator3D` 只接受 ASTRA regular circular `cone` geometry 和 cubic voxels；
- `parallel3d`、任意 `cone_vec` 和非等边体素不会被伪装成已支持；
- ASTRA FDK 使用官方 `FDK_CUDA`，支持 Ram-Lak、Parker short-scan weighting 和
  voxel supersampling；
- 128×128 tissue case 和 32×32 Shepp-Logan case 是确定性合成 phantom，不是患者数据；
- 大型或受许可约束的数据集只登记在 `test/data/external_catalog.json`，不会自动下载。

外部数据与可选后端：

- [2DeteCT](https://zenodo.org/records/6802615)：CC BY 4.0；
- [FIPS Walnut CT](https://fips.fi/dataset.php)：使用时遵守数据集条款；
- [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox)：GPL-3.0；
- [LLNL LEAP](https://github.com/LLNL/LEAP)：MIT。

## 许可证

项目元数据声明为 MIT。外部后端和数据集保持各自许可证与使用条款。
