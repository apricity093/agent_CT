> MRI / deconvolution：各提供七个可执行算法，包含 VarNet、DCP 与 L0 盲反卷积。VarNet 需要显式匹配权重。见[公开 API](docs/agent_algorithm_api.md)和[外部输入示例](docs/public_benchmarks.md)。

MRI 与反卷积的主展示和质量基准现使用[标准轻量公开数据](docs/public_benchmarks.md)：
camera、astronaut、cell，Levin09/Airy 核，以及标准 MR Shepp–Logan 组织模体与
环形线圈编码。运行 `tools/run_public_benchmarks.py`，结果见
[标准基准实测](outputs/public_benchmarks/)。算法与数据对应关系见[公共基准](docs/public_benchmarks.md)。
真实 fastMRI 膝关节 VarNet 评价仍需用户提供获授权的本地采集，不包含在模拟模体分数中。

> CT 接口与验收：见 [公开 API](docs/agent_algorithm_api.md) 和 [验证说明](docs/validation.md)。

# inv_framework

`inv_framework` 是一个面向 X-ray CT 重建与评测的轻量 Python 框架。它将 CT
前向/伴随算子、经典重建算法、版本化测试数据、参数验证、收敛诊断和固定参数评测
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

| CLI 名称 | 算法 | 几何 | 主要配置 | `invct run` 后端 |
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

### 公开任务 API 的后端支持

上表列出 `invct run` 使用的后端。通过 `run_ct_task()` 或 `invct task` 提交请求时，
可用 `CTTaskRequest.backend` 选择后端，用 `device` 选择 CPU/CUDA；二维默认使用
`backend="torch"`、`device="cpu"`。`invct run` 没有 `--backend` 参数。

| `backend` | 任务几何 | 设备与依赖 | 验证状态 |
| --- | --- | --- | --- |
| `torch` | `parallel_2d` | PyTorch CPU/CUDA | 已执行二维重建与伴随检查 |
| `astra` | `parallel_2d` | ASTRA；CUDA 路径还需 PyTorch CUDA 与 ASTRA CUDA | CPU/CUDA 伴随检查及资源释放检查已通过 |
| `astra` | `cone_3d`，FDK | PyTorch CUDA 与 ASTRA CUDA | spheres 回归与 tissue FDK 展示已执行 |
| `leap` | `parallel_2d` | 可导入的 LLNL LEAP `leapctype`；设备取决于其安装配置 | 任务接口已接入，尚未完成真实 LEAP 后端验收 |

ASTRA 二维 CUDA 路径在 GPU 上应用 ASTRA 线性投影矩阵及其转置。
`list_capabilities()` 返回各后端的依赖、设备与可运行信息，以及算法的默认参数和兼容性要求；
依赖可用不等于完成数值验收。后端切换仍需满足算法、几何与观测域约束。
完整验证范围与未通过的回归见[验证说明](docs/validation.md)。

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

默认数据根目录为 `tests/data`。查询 catalog：

```bash
invct data list
invct data list --dimension 2
invct data list --geometry parallel_2d --tag dense_clean
invct data show parallel_2d__ellipses_dense_clean
```

新二维案例的场景标签为 `dense_clean`、`sparse_gaussian`、`sparse_poisson` 和
`limited_angle`。过滤后没有匹配案例时，表格仅显示表头，`--json` 返回 `[]`，退出码为 `0`。

检查案例元数据与 HDF5 数组形状：

```bash
invct data validate
invct data validate parallel_2d__ellipses_dense_clean
```

外部 catalog 可通过 `--root PATH`（数据子命令）或 `--data-root PATH`（`run`）传入。
路径指向包含 `catalog.json` 的目录，例如：

```bash
invct data list --root /path/to/ct-data --json
invct data show parallel_2d__ellipses_dense_clean --root /path/to/ct-data
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
  --case parallel_2d__ellipses_dense_clean \
  --config configs/algorithms/cgls.yaml \
  --out artifacts/cgls_dense_cpu \
  --device cpu
```

CUDA 示例：

```bash
invct run tv_fista \
  --case parallel_2d__ellipses_sparse_poisson \
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

输出目录非空时命令默认拒绝运行，防止混合不同运行的输出。只有确认替换该次 run
产物时才显式添加 `--overwrite`。

### 选择 solver 与正则项

`invct run` 的位置参数选择 solver，YAML 的 `parameters.regularizer` 选择受支持的
正则项；当前 CLI 没有 `--regularizer` 或 `--operator` 参数。已有组合示例：

```bash
invct run fista --case parallel_2d__ellipses_dense_clean --config configs/examples/fista_haar_l1.yaml --out artifacts/fista_haar_cpu --device cpu
invct run tikhonov --case parallel_2d__ellipses_dense_clean --config configs/examples/tikhonov_h1.yaml --out artifacts/tikhonov_h1_cpu --device cpu
```

直接重建不要求迭代 solver，无正则算法不要求 regularizer。CLI/runtime 的 CT operator
由实际 case geometry 构建；其他 Python operator 类的存在不代表它们已经接入 CLI。
几何、投影数和观测噪声的变化需要相应输入数据，不能只修改算法参数或描述字段。

### 运行预算与停止策略

```bash
invct run fista --case parallel_2d__ellipses_dense_clean --config configs/algorithms/fista.yaml --out artifacts/fista_budget_cpu --device cpu --max-iterations 50
```

| 选项 | 含义 |
| --- | --- |
| `--max-iterations N` | 外层迭代预算；算法迭代数仍由配置指定并受预算限制 |
| `--stopping-policy PATH` | schema `1.0`、mode `solver_native_and_endpoint` 的 JSON 停止策略 |
| `--parameter-sources PATH` | 参数名到来源标签的 JSON mapping |
| `--observation-domain DOMAIN` | 选择 case 中已提供的观测域，不生成新的观测数据 |
| `--fixed-compute` | 关闭 solver 提前停止，用于声明了固定计算预算的实验 |

`--fixed-compute` 不取消预算或数值异常检查。Agent 取 `budget.max_wall_seconds` 与
`budget.timeout_s` 中已指定的最小值作为 worker 的进程期限。达到迭代上限或预算耗尽不表示收敛。
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

当前 suite 使用固定默认参数，记录质量、数据一致性和运行时间。数据与输出格式见 [数据集说明](docs/datasets.md)。

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
[Agent 用法](docs/agent_algorithm_api.md)。

完整执行路径为：

```text
RunRequest
  -> Agent candidate selection and compatibility filtering
  -> Agent parameter/budget validation
  -> CTBackend / CTAdapter
  -> CT worker / inv_framework.api.run_ct_task
  -> CT solver and native diagnostics
  -> Independent evaluator
```

Agent 负责问题解释、候选选择、参数来源、尝试谱系和跨运行比较；CT backend 负责数值
合法性、solver-native 停止判据及数值诊断。达到最大迭代数或预算
耗尽不会被 Agent 改写为 `converged`。

## 输出与诊断

单次成功运行生成：

```text
run-directory/
├── reconstruction.pt
├── metrics.json
├── diagnostics.json
└── comparison.png
```

`reconstruction.pt` 保存 CPU tensor bundle，便于在没有原运行 GPU 的机器上离线复评。
`diagnostics.json` 记录算法参数、参数来源、停止原因和运行时间。失败运行写入结构化 failure 文件。

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
运行时间。

L2 数据项下，Landweber 的固定步长条件为 `0 < step_size < 2 / ||A||^2`，
FISTA 的标准固定步长条件为 `0 < step_size <= 1 / ||A||^2`。省略步长时使用
确定性有限 power iteration 估计；该估计不是谱范数上界。当前 FISTA/TV-FISTA
对试探步长检查实际光滑项 majorization，失败时减半回溯。有限 TV proximal 的限制见组合算法文档。

## 可组合算法与 Agent

当前完整支持矩阵及数学、数据和参数要求见 [已支持的 CT 组件组合](docs/algorithms.md)。

CT 请求将 `solver`、`operator`、`regularizer`、`data_fidelity`、
`observation_model`、`constraints` 与公开 `noise_parameters` 分开表达。
Agent 先发现候选，再校验物理域、数据字段和数学兼容性，最后装配并执行。
已登记的二次组合可使用 L2、已知方差 Gaussian NLL 或固定 WLS；
FISTA/ADMM/PDHG 还提供纯透射 Poisson NLL 路径。正则包括无正则、恒等
Tikhonov、H1、两种 TV、梯度 Huber 与 Haar detail-L1。
这表示受兼容性约束的组件组合，并不表示任意 solver、prior 与几何均可互换；
执行完成、优化收敛和重建质量分别记录。

参数、数学条件、约束 TV 和来源许可见[CT 算法](docs/algorithms.md)。
任务字段和停止语义见[公开 API](docs/agent_algorithm_api.md)，测试入口及限制见
[验证说明](docs/validation.md)。

## 已知精度与成本限制

约束 PDHG 依赖凸 TV 与 A/D 谱上界，必须同时满足可行性和原生 fixed-point
条件才报告 converged。ASD 为 SART2D feasibility-backtracking 变体，无全局最优保证。
`tv_fista` 的 TV proximal 为有限 FGP；通用 `fista` 的 TV/受约束 Haar
使用带 gap 及误差目标的联合近端，内层精度不足不会标为收敛。

## Python API

### 公开任务接口

Agent 和普通重建任务使用 `inv_framework.api`。`list_capabilities()` 返回算法、
默认参数、保真项、正则项、后端与案例列表；`run_ct_task()` 完成请求校验、观测域适配、
算子与求解器装配、重建和指标计算，并在结束时释放所持有的 ASTRA 资源。

```python
from inv_framework.api import CTTaskRequest, list_capabilities, run_ct_task

capabilities = list_capabilities()
print([item["name"] for item in capabilities["algorithms"]])
print([item["case_id"] for item in capabilities["cases"]])

request = CTTaskRequest(
    case_id="parallel_2d__ellipses_sparse_poisson",
    algorithm="pdhg",
    backend="torch",
    device="cpu",
    fidelity="transmission_poisson_nll",
    regularizer="tv",
    params={"num_iterations": 2, "strength": 0.01, "min_value": 0.0},
)
result = run_ct_task(request)
if result.status == "failed":
    raise RuntimeError(result.error_message)
print(result.convergence_status, result.iterations, result.metrics)
reconstruction = result.reconstruction  # NumPy 数组；本例形状为 (1, 1, 64, 64)
```

未指定的算法参数从 `configs/algorithms/<algorithm>.yaml` 加载；外部数据通过
`data_root` 指定包含 `catalog.json` 的目录。示例的两次迭代用于验证调用链路，
不代表优化收敛。应同时读取 `status`、`converged`、`convergence_status` 和 `reason_code`；
有真值时返回图像质量指标，无真值时仍可重建并报告数据一致性指标。

等价字典请求也可直接传给 `run_ct_task()`，或保存为 JSON 后执行：

```bash
invct task request.json --out artifacts/ct_task
```

该命令保存 `reconstruction.npy`（有重建结果时）和 `diagnostics.json`。
前文的 `reconstruction.pt` 等文件是 `invct run` 的输出格式。
请求字段、取消机制与结果语义见[公开 API](docs/agent_algorithm_api.md)。

### 底层算子与求解器

需要自行装配数值算法时，可直接使用以下接口；本例与默认 Torch CLI 使用相同的 case 和 operator：

```python
import torch

from inv_framework.benchmarks import load_ct_case
from inv_framework.ct.backends import ParallelBeamRadon2D
from inv_framework.solvers import CGLSSolver

device = "cuda" if torch.cuda.is_available() else "cpu"
case = load_ct_case(
    "parallel_2d__ellipses_dense_clean",
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
ct-benchlab/
├── inv_framework/
│   ├── api/         # schema、dispatcher、execution、CLI 和文件输出
│   ├── core/        # operator、fidelity、regularizer
│   ├── ct/          # 案例、测量、registry 与 backends
│   ├── solvers/     # analytic、algebraic、subset、statistical、splitting、deep
│   └── utils/       # 原始数值指标与三维正交切片
├── configs/         # 算法默认值、基准和评价协议
├── tests/           # 唯一测试目录，含四个 ellipses 和 spheres 3D
├── docs/
├── examples/
└── tools/
```

## 开发与测试

安装项目声明的开发依赖后，默认回归仅收集 `tests/`：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

CI 使用同一入口。CUDA/ASTRA 与真实 LEAP 测试按依赖、硬件和模型配置跳过，
CPU 环境可以运行基础测试。400 次 DIP 训练不在默认集合中，按需显式运行：

```bash
python -m pytest -q -m training tests/test_nonlinear_operator.py
```

完整 benchmark 脚本保留在 `tests/ct_all_solvers/run_all_solvers.py`，
FDK 模拟脚本保留在 `tests/ct_fdk_simulation/run_fdk_simulated_smoke.py`；
二者通过脚本命令显式运行，不由 pytest 自动执行。

快速验证新数据、存储与重建入口：

```bash
python -m pytest -q tests/test_ellipses_runtime.py tests/ct_cases
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
README 中的 `configs/examples`、benchmark/protocol 文件、`tests/data` 和 docs
按源码 checkout 使用，不随 wheel 提供；安装包运行时需要显式提供数据目录和配置。
固定版本安装必须使用已推送且可获取的 commit，再同步主项目依赖引用，不能用
本地 dirty checkout 的结果代替固定来源安装验收。

## 能力边界与数据许可

- `ASTRAFDKOperator3D` 只接受 ASTRA regular circular `cone` geometry 和 cubic voxels；
- `parallel3d`、任意 `cone_vec` 和非等边体素不会被伪装成已支持；
- ASTRA FDK 使用官方 `FDK_CUDA`，支持 Ram-Lak、Parker short-scan weighting 和
  voxel supersampling；
- 二维 ellipses、三维 tissue 与 spheres 案例均为合成体模，不是患者数据；
- 大型或受许可约束的数据集只登记在 `tests/data/external_catalog.json`，不会自动下载。

外部数据与可选后端：

- [2DeteCT](https://zenodo.org/records/6802615)：CC BY 4.0；
- [FIPS Walnut CT](https://fips.fi/dataset.php)：使用时遵守数据集条款；
- [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox)：GPL-3.0；
- [LLNL LEAP](https://github.com/LLNL/LEAP)：MIT。

## 许可证

项目元数据声明为 MIT。外部后端和数据集保持各自许可证与使用条款。

MRI 与反卷积的完整算法、数据和任务入口见[公共基准](docs/public_benchmarks.md)。
