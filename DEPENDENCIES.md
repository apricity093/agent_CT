# 依赖与安装

本文是 `inv-framework` 数值仓库的统一依赖说明，覆盖 CT、MRI、deconvolution、
数据准备和开发验证。以下命令均在本仓库根目录执行；在上层 Agent 项目中，
应先进入 `external/ct-benchlab`。本包可以独立安装，不需要 Iteris、USCT 或 Agent SDK。

## 推荐安装环境

本次交付推荐使用 **Python 3.12**。Windows / Python 3.12 的独立环境已验证
三域能力查询、默认配置读取、公开 CLI、三个 CPU 示例和用户自有输入；
模块与配置从 site-packages 加载，pip check 通过。

已验证的主要依赖组合如下：

| 包 | 已验证版本 | 该版本声明的 Python 最低要求 |
| --- | --- | --- |
| NumPy | 2.5.2 | 3.12 |
| SciPy | 1.18.0 | 3.12 |
| PyTorch | 2.5.1 | 3.8 |
| h5py | 3.16.0 | 3.10 |
| PyYAML | 6.0.3 | 3.8 |
| Matplotlib | 3.11.1 | 3.11 |

其中 NumPy 和 SciPy 决定这组依赖不能安装在 Python 3.10/3.11 上。
上述要求来自对应版本的包元数据，不代表任意更高 Python 版本均已验证。

在 Windows PowerShell 中，从算法库根目录使用已安装的 Python 3.12 创建新环境：

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install . "numpy==2.5.2" "scipy==1.18.0" "torch==2.5.1" "h5py==3.16.0" "PyYAML==6.0.3" "matplotlib==3.11.1"
& ./.venv/Scripts/python.exe -m pip check
```

若未安装 Windows Python Launcher，将 `py -3.12` 替换为 Python 3.12
解释器的绝对路径。Linux/macOS 可使用 `python3.12` 创建环境和
`.venv/bin/python` 安装，但这两个平台未纳入本次验收。
以上命令固定主要依赖，其他传递依赖仍由 pip 正常解析；它不是完整环境锁定。

示例输入和配置保留在 [examples/minimal_cpu](examples/minimal_cpu/)，
安装与 CPU 验证不依赖测试案例目录、旧 Agent、ASTRA 或下载权重。
本次安装验证不代表完整 CT 数值、伴随、3D FDK 或 GPU 显存验收通过。

## 基础依赖范围

[pyproject.toml](pyproject.toml) 中的 `requires-python = ">=3.10"` 是包允许解析安装的最低声明，
不是 Python 3.10/3.11 已通过验收的承诺。该声明配合下表的宽泛依赖范围，
允许 pip 为不同 Python 选择相容的依赖版本；这些组合并未逐一验证。
需要采用本次已验证组合时，请使用上文的 Python 3.12 安装命令。
可选功能还须满足各自依赖与运行时要求。

| 包 | 声明版本 | 用途 |
| --- | --- | --- |
| `torch` | `>=1.12` | CT 算子与求解器、深度先验、MRI VarNet/NUFFT |
| `numpy` | `>=1.21` | 数组、MRI 与反卷积计算及文件输入输出 |
| `scipy` | `>=1.8` | 数值线性代数、信号处理和稀疏计算 |
| `h5py` | `>=3.7` | HDF5 数据与 fastMRI 切片准备 |
| `PyYAML` | `>=6.0` | 实验与算法配置 |
| `matplotlib` | `>=3.5` | 结果绘图与展示 |

构建依赖为 `setuptools>=68` 和 `wheel`，由 pip 构建流程处理。

以下通用命令只按声明范围解析依赖，不保证得到上文的已验证版本组合：

```bash
python -m pip install .
```

开发时使用可编辑安装并加入 pytest：

```bash
python -m pip install -e ".[dev]"
```

这两种安装均不包含 ASTRA、LEAP、非笛卡尔 MRI 扩展或模型权重，
也不保证安装的 PyTorch 带 CUDA 支持。

## 按功能选择依赖和设备

| 领域／功能 | 基础依赖之外的要求 | 当前设备支持 |
| --- | --- | --- |
| CT 二维：`fbp`、`sirt`、`cg`、`landweber`、`cgls`、`lsqr`、`sart`、`os_sart`、`mlem`、`osem`、`tikhonov`、`tv_fista`、`fista`、`admm`、`pdhg`、`constrained_pdhg`、`asd_pocs_tv`，使用 `torch` 后端 | 无；GPU 运行需 CUDA 版 PyTorch | CPU / CUDA |
| 同上，使用 `astra` 后端 | `astra-toolbox`；GPU 路径同时要求 PyTorch CUDA 和 ASTRA CUDA | CPU / CUDA |
| CT 三维锥束 `fdk` | `astra-toolbox`，且 PyTorch CUDA 与 ASTRA CUDA 均可用 | 仅 CUDA |
| CT 二维 `leap` 后端 | LLNL LEAP 的 `leapctype` 模块及其本机动态库 | 取决于 LEAP 构建与配置；CUDA tensor 另需 PyTorch CUDA |
| MRI 笛卡尔：`zero_filled`、`cg_sense`、`cs_tv`、`cs_wavelet`、`grappa`、`espirit` | 无 | CPU |
| MRI 非笛卡尔：`cg_sense`、`cs_tv`、`cs_wavelet` | `.[mri]`，即 `torchkbnufft==1.5.2` | CPU |
| MRI `varnet` | 显式提供与网络配置匹配的本地 checkpoint；GPU 运行需 CUDA 版 PyTorch | CPU / CUDA；仅二维笛卡尔 |
| 反卷积：`wiener_hunt`、`richardson_lucy`、`tv_admm`、`rl_tv`、`hyper_laplacian`、`blind_dcp`、`blind_l0` | 无 | CPU |
| CT 库级 DIP / INR | 内置模型使用基础依赖；自定义模型由调用方提供 | 随输入 tensor 与算子设备 |
| CT 库级 DPS | 显式配置模型与噪声调度器；使用 Diffusers 适配器时安装 `.[diffusion]` | 随模型、tensor 与算子设备 |
| MRI／反卷积公共基准数据生成 | `.[benchmark-data]` | CPU |

MRI 和反卷积另有 `admm`、`pdhg`、`fista`、`landweber`、`cgls`、`lsqr`、`cg` 的共享二维 CPU 入口，无基础依赖之外的要求。三域范围与限制见[组合说明](docs/composition.md)。

MRI 和反卷积不使用 ASTRA。VarNet 网络在仓库内提供，不需要另外安装
`fastmri` 或 PyTorch Lightning；权重不随包分发，执行时不会自动下载。
DIP、INR、DPS 是库级接口，不属于 CT 公开任务 API 的 18 个算法。

CT 设备与后端是两个独立选项。`run_ct_task()` / `invct task` 的二维任务默认
`backend="torch"`、`device="cpu"`；三维 FDK 请求使用 `backend="astra"`、
`device="cuda"`。`invct run` 没有 `--backend` 选项，二维使用 PyTorch，FDK 使用 ASTRA。
本项目 FDK 调用 ASTRA 的 `FDK_CUDA`，没有 CPU 回退实现。
二维 ASTRA CUDA 路径用 GPU 稀疏矩阵乘法应用 ASTRA 线性投影矩阵及其转置。

## Python 可选依赖

| extra | 声明的额外包 | 安装命令 |
| --- | --- | --- |
| `dev` | `pytest>=7`、`scipy` | `python -m pip install -e ".[dev]"` |
| `mri` | `torchkbnufft==1.5.2` | `python -m pip install -e ".[mri]"` |
| `benchmark-data` | `scikit-image==0.26.0`、`phantominator==0.7.0` | `python -m pip install -e ".[benchmark-data]"` |
| `diffusion` | `diffusers>=0.30` | `python -m pip install -e ".[diffusion]"` |

可组合安装，例如开发、非笛卡尔 MRI 和公共基准生成：

```bash
python -m pip install -e ".[dev,mri,benchmark-data]"
```

`benchmark-data` 用于 `tools/generate_public_benchmarks.py`；读取已有 NPY/NPZ
任务不需要这些生成器。运行包含非笛卡尔 MRI 的公共基准还需要 `mri`。
准备公开核和部分 scikit-image 图像时需要联网，详见[公共基准](docs/public_benchmarks.md)。
模型 checkpoint、真实 fastMRI 采集和基准数据文件属于独立输入，不由 pip 安装提供。
CT catalog 数据默认来自源码中的 `tests/data`，不随 wheel 分发；安装包执行 catalog
任务时需提供数据目录，见[任务 API](docs/agent_algorithm_api.md)。

## CUDA、ASTRA 与 LEAP

### PyTorch CUDA

需要 NVIDIA GPU、兼容所选 CUDA 运行时的驱动及 CUDA 版 PyTorch。
按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
选择操作系统、Python 环境和 CUDA 构建，先安装该版本的 `torch`，再安装本包。
本仓库没有直接依赖 `torchvision` 或 `torchaudio`。
包声明中的 `torch>=1.12` 不指定 CPU/CUDA 构建，也不代表支持当前所有 GPU 架构。

检查当前解释器：

```bash
python -c "import torch; print('torch:', torch.__version__, 'runtime:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

### ASTRA

ASTRA 未列入基础依赖或 extras，需要单独安装。Windows/Linux 的 Conda 环境可按
[ASTRA 官方安装说明](https://astra-toolbox.com/docs/install.html)安装：

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

也可按同一官方页面选择适合当前 Python、平台和驱动的 pip 包：

```bash
python -m pip install astra-toolbox
```

这两种方式择一；包的 CUDA 运行时要求随版本变化，不能仅凭 `import astra` 成功
判断 FDK 可用。FDK 和二维 ASTRA CUDA 任务要求以下两个可用性检查都为 `True`：

```bash
python -c "import torch; import astra; print('PyTorch CUDA:', torch.cuda.is_available()); print('ASTRA CUDA:', astra.use_cuda())"
```

仅运行二维 ASTRA CPU 任务时不要求 CUDA。使用 CPU-only ASTRA 构建的方法见
上述官方安装页。当前项目已有 ASTRA 2.2.0 与 PyTorch 2.5.1 / CUDA 12.1 的
Windows 验证记录；这是已验证组合，不是所有平台通用的版本锁定。

### LEAP

按 [LLNL LEAP 官方说明](https://github.com/LLNL/LEAP)安装 Python 绑定和本机库，
确保同一 Python 环境能创建投影模型：

```bash
python -c "from leapctype import tomographicModels; model = tomographicModels(); print(type(model).__name__)"
```

本仓库没有 `leap` extra。CPU/GPU 能力由所安装的 LEAP 构建与配置决定。
当前公开任务只绑定二维平行束，FDK 仍使用 ASTRA；LEAP 真实后端验收尚未完成。

## Windows Conda 与验证入口

激活环境后，在 PowerShell 中设置该环境的 DLL 路径及线程数：

```powershell
$env:PATH = "$env:CONDA_PREFIX/Library/bin;$env:CONDA_PREFIX;" + $env:PATH
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
```

安装后检查依赖解析和命令入口：

```bash
python -m pip check
invct --help
inv-run --help
```

CT 后端发现可通过 `from inv_framework.api import list_capabilities` 查询。
MRI／反卷积的 `list_capabilities()` 描述算法契约，不代替可选依赖、权重和实际任务检查。
导入、设备检查成功也不代表数值验收通过；测试命令和已知限制见
[验证说明](docs/validation.md)。开发依赖不会自动启用 GPU、LEAP 或外部权重测试。
