# AGENTS.md

## 项目主要任务

本项目的主要任务是：将经典 CT 求解算法按照当前项目已经存在的模板、规范和接口进行改写与接入，使其成为 `inv_framework` 风格的 CT 重建能力。

后续开发必须以当前项目现有实现为准，不应另起一套不兼容的接口或目录结构。涉及经典 CT 求解算法时，应优先遵守：

- 算子侧使用项目已有的 `ForwardOperator` / `LinearOperator` 抽象表达 CT 正投影、反投影、伪逆和投影约束等能力。
- 求解器侧使用项目已有的 `InverseProblemSolver` 风格接口，围绕 `solve(y, operator, **kwargs)` 组织重建流程。
- CT 算法改写应贴合当前项目已有 solver wrapper、函数式 solver、子包导出和示例/测试组织方式。
- 经典 CT 算法的数值含义、初始化策略、迭代结构、停止条件和后端能力边界应保持清晰；不能为了适配接口而含糊改变算法语义。
- 如果某个经典 CT 算法依赖当前项目尚未具备的后端能力，应显式做后端门控或新增隔离适配层，不应把未完成能力伪装成完整实现。

本主要任务必须与下方“Python 既有实现冻结规则”同时遵守。也就是说，后续可以为 CT 算法适配新增文件、新增隔离实现或追加导出；但不得在未获用户明确许可的情况下改写 2026-06-08 已经存在的 `.py` 实现基线。

## 当前 Python 既有实现冻结规则

本文件记录本 `AGENTS.md` 所在目录及其子目录下所有 `.py` 文件在 2026-06-08 已经存在的实现基线。冻结对象是这些文件中当前已经存在的实现、接口和行为，不是把整个 `.py` 文件永久设为只读。后续任何改动都必须遵守：

- 不允许改写、删除、重命名或移动本文件列出的任何既有类、函数、方法、导入导出、默认参数、控制流、数值逻辑、异常行为或占位实现。
- 允许在本文件列出的 `.py` 文件中增加新内容，例如新增类、函数、solver wrapper、辅助工具、测试用例、示例代码、导入项或导出项；新增内容必须与既有实现清晰区分，并且不得改变既有实现的行为。
- 如新增功能需要扩展 `__init__.py` 的公开导出，只能追加新导出，不能移除、改名或改变当前已有导出含义。
- 不允许为了新增功能而顺手调整这些文件中的既有实现；如果确实需要修改既有实现才能完成任务，必须先停止并向用户说明原因，等待用户明确覆盖本规则。
- SHA256 用作 2026-06-08 原始实现快照的校验基线。后续如果只是合法新增内容，整个文件的 SHA256 可能变化；此时应通过 diff 判断是否只增加了内容，不能把全文件哈希变化本身直接视为违规。

基线生成日期：2026-06-08。

## Python 既有实现基线清单

| 文件 | 行数 | 字节 | SHA256 | 现有实现摘要 |
| --- | ---: | ---: | --- | --- |
| `examples/ct_demo.py` | 187 | 7614 | `1e7498553fff19dd2f7f180f107fff7c3a7605a922e51fb08dcd27ba56c51fef` | CT 示例脚本，生成 Shepp-Logan phantom 并演示 Radon 投影、重建流程和结果展示。 |
| `examples/nonlinear_demo.py` | 156 | 6080 | `151636f8974f85cee607e998b9ed13f8d7fbd9319d1c9f6208441a969a1ecff3` | 非线性反问题示例，定义饱和 Radon 算子并演示非线性求解流程。 |
| `examples/train_diffusion_for_dps.py` | 132 | 4889 | `dd6716374c884b714a863eb73363f6df18825537bedf98616d7b718294110805` | DPS 扩散模型训练示例，生成随机 phantom batch 并构建/训练 tiny diffusion 模型。 |
| `inv_framework/__init__.py` | 21 | 856 | `3ffb96874e3fb6bc7066388a64863697db1c521220ca3c1349ec235958bbb9ea` | 包级公开导出入口，重导出算子、噪声模型、求解器、模型和指标组件。 |
| `inv_framework/models/__init__.py` | 5 | 158 | `210e38d70f22e7e120d142099b70d7392a27c0c5647c9e307bdad9d8f124e5ba` | models 子包导出入口，公开 SIREN、SkipUNet 和 TinyUNet。 |
| `inv_framework/models/siren.py` | 54 | 2056 | `45c666f5d6299847668b6f97f83467e0a25e4e83ddbec1885dea1adc30bdd9c9` | 定义 SIREN 隐式神经表示网络及其 sine layer。 |
| `inv_framework/models/skip_unet.py` | 53 | 1859 | `dbdabb4cbf486f2de6cd93e371ee4ace27c1d0fb436ca185f7419064153e2a03` | 定义 DIP 风格 SkipUNet 和卷积块。 |
| `inv_framework/models/tiny_unet.py` | 71 | 2825 | `4407ff4751fadfc539816561ca210cb24ab6bf91b23b0d62cd985b060fc934bf` | 定义时间步嵌入、时间条件卷积块和 TinyUNet 扩散模型。 |
| `inv_framework/operators/__init__.py` | 11 | 270 | `42a91d969501f3d46829793c13e4273d544797ac8d3d0474c5aafd4385887ea3` | operators 子包导出入口，公开基础算子和噪声模型。 |
| `inv_framework/operators/base.py` | 74 | 2693 | `fd5ebc62c8013c2c3b9ab403e2b6a578d769afb13761afc57bf1be1aee72db34` | 定义 `ForwardOperator` 与 `LinearOperator` 抽象接口、默认 `pseudo_inverse` 和 `project` 行为。 |
| `inv_framework/operators/ct/__init__.py` | 9 | 214 | `64fdbf36317a7346a2772a8bdc03658706e39080ded357cfaf68e27bc6c150e7` | CT 算子子包导出入口，公开 2D torch Radon 和 3D ASTRA 适配器。 |
| `inv_framework/operators/ct/astra_adapter.py` | 120 | 4457 | `0006e36bf7c689c5e91e973ba5a822c2a45fde3ff726224a0627501f6e27703d` | 定义 ASTRA 后端依赖检查、autograd bridge 和 `ASTRAOperator3D` 接口。 |
| `inv_framework/operators/ct/radon_torch.py` | 146 | 5216 | `902333d72d6fa4f7a6e67ef82e48f9e69a351e292b0fa8a6d78caeb1a53d21b5` | 实现 2D parallel-beam Radon torch 算子，包括旋转网格、forward/adjoint autograd 和 pseudo-inverse。 |
| `inv_framework/operators/noise.py` | 55 | 1690 | `e5426a2dbb2d7a3f1fc6540554a52edfba1e879ccc8b5fd98b5bb7fc085c8680` | 定义噪声模型基类、无噪声、高斯噪声和 log-domain Poisson 噪声。 |
| `inv_framework/solvers/__init__.py` | 16 | 352 | `442d5bacdc5817b6c5395debc24a1426fa6d34d6b7caf0c687c7c7fdd29d3235` | solvers 子包导出入口，公开 classical、DIP、INR 和 diffusion 求解器。 |
| `inv_framework/solvers/base.py` | 24 | 768 | `88792024c363adc08bfc550ae2bbfee8537d9117b192b6f17d3c37e58bb12868` | 定义 `InverseProblemSolver` 抽象接口和 `solve(y, operator, **kwargs)` 合约。 |
| `inv_framework/solvers/classical.py` | 161 | 5604 | `ebd77389b59657096d1eb317a8e78b27150e0d19085fbbc537bebc83d7de5096` | 实现 FBP、SIRT、Landweber 经典线性求解器函数及对应 solver wrapper。 |
| `inv_framework/solvers/diffusion/__init__.py` | 20 | 476 | `ff59a91fe3cee6fb5b41b8818b8ca5a7048257146fd9c0b224738a39149a0ed3` | diffusion 子包导出入口，公开调度器、条件方法、DPS 和 diffusers 兼容组件。 |
| `inv_framework/solvers/diffusion/conditioning.py` | 59 | 2138 | `954dafc75d6d5f1bfc094753735f8c954f6ac24879d7d31929f8a9e12bf3d62b` | 定义 diffusion 条件更新接口和 posterior sampling 条件方法。 |
| `inv_framework/solvers/diffusion/diffusers_compat.py` | 122 | 4389 | `46df849798ab4b531fc36b6ceadc10b42ddfecb483cfce42a865fe3d5aa1774b` | 定义 diffusers scheduler 适配器和 UNet wrapper。 |
| `inv_framework/solvers/diffusion/dps.py` | 93 | 3569 | `7ce9baf176f83fc024303e2594a3ca1f408b004fd41c96a74245de9ae7dea5f5` | 定义 `DPSSolver`，实现 diffusion posterior sampling 求解流程。 |
| `inv_framework/solvers/diffusion/scheduler.py` | 120 | 4513 | `2df36deee593bdb683f00b61bb52e88d6ed254e70a96bd41512c336be56aeaaf` | 定义 noise schedule 抽象接口和 VP schedule 实现。 |
| `inv_framework/solvers/dip.py` | 79 | 2663 | `07a0a51914b8487cdbd5210ed5513e29067c67a328510b46c7e405b12b61026b` | 定义 `DIPSolver`，通过网络参数优化求解反问题。 |
| `inv_framework/solvers/inr.py` | 72 | 2526 | `640c61a74d4da9b03501dc3a38aae99f62cbf91678617ef2c395985cb099eac6` | 定义 `INRSolver`，通过隐式神经表示优化求解反问题。 |
| `inv_framework/utils/__init__.py` | 3 | 60 | `c6e1ebcf5553249a375f8343394f9d28451a7f1c9aed6c5a5284f0aea60e741d` | utils 子包导出入口，公开图像质量指标函数。 |
| `inv_framework/utils/metrics.py` | 36 | 1543 | `b21cf4b20cf817de56d0876ae7c6d6a1781d340f6bee43a0bf1ce3ef7e03e54b` | 定义 PSNR 和 SSIM 指标函数。 |
| `tests/test_nonlinear_operator.py` | 94 | 3511 | `bb00bab12c401c5bf701c46be8081a07c3ccc114ab89eef7615f091e294cd71a` | 测试非线性算子合约、classical solver 拒绝非线性输入和 DIP 非线性恢复。 |
| `tests/test_radon_adjoint.py` | 69 | 2703 | `acfa6ca9777b5ac22846adee648e1ef03e306c3bf46a7e10c5097ae7f4fbb0cf` | 测试 Radon autograd/adjoint 一致性和 FBP disk phantom 恢复。 |

## 符号冻结索引

### `examples/ct_demo.py`

- `shepp_logan` at line 33
- `main` at line 63

### `examples/nonlinear_demo.py`

- `SaturatedRadon2D` at line 35: `__init__`, `forward`, `invert_saturation`
- `main` at line 68

### `examples/train_diffusion_for_dps.py`

- `random_ellipse_phantom` at line 30
- `sample_batch` at line 52
- `build_tiny` at line 57
- `build_diffusers` at line 63
- `main` at line 82

### `inv_framework/models/siren.py`

- `SineLayer` at line 8: `__init__`, `forward`
- `SIREN` at line 25: `__init__`, `forward`

### `inv_framework/models/skip_unet.py`

- `_ConvBlock` at line 8: `__init__`, `forward`
- `SkipUNet` at line 24: `__init__`, `_up`, `forward`

### `inv_framework/models/tiny_unet.py`

- `timestep_embedding` at line 13
- `_TimeBlock` at line 23: `__init__`, `forward`
- `TinyUNet` at line 40: `__init__`, `forward`

### `inv_framework/operators/base.py`

- `ForwardOperator` at line 25: `forward`, `__call__`
- `LinearOperator` at line 47: `adjoint`, `pseudo_inverse`, `project`

### `inv_framework/operators/ct/astra_adapter.py`

- `_require_astra` at line 24
- `_ASTRAFunction` at line 32: `forward`, `backward`
- `ASTRAOperator3D` at line 74: `__init__`, `forward`, `adjoint`

### `inv_framework/operators/ct/radon_torch.py`

- `_rotation_grid` at line 15
- `_radon_forward_impl` at line 32
- `_radon_adjoint_impl` at line 51
- `_RadonFunction` at line 73: `forward`, `backward`
- `_RadonAdjointFunction` at line 87: `forward`, `backward`
- `ParallelBeamRadon2D` at line 101: `__init__`, `forward`, `adjoint`, `pseudo_inverse`

### `inv_framework/operators/noise.py`

- `NoiseModel` at line 8: `forward`
- `NoNoise` at line 16: `forward`
- `GaussianNoise` at line 21: `__init__`, `forward`
- `PoissonLogDomainNoise` at line 30: `__init__`, `forward`

### `inv_framework/solvers/base.py`

- `InverseProblemSolver` at line 9: `solve`

### `inv_framework/solvers/classical.py`

- `_require_linear` at line 22
- `_ram_lak` at line 31
- `_filter_sino` at line 40
- `fbp` at line 51
- `sirt` at line 65
- `landweber` at line 93
- `FBPSolver` at line 119: `__init__`, `solve`
- `SIRTSolver` at line 127: `__init__`, `solve`
- `LandweberSolver` at line 144: `__init__`, `solve`

### `inv_framework/solvers/diffusion/conditioning.py`

- `ConditioningMethod` at line 16: `__init__`, `apply`
- `PosteriorSampling` at line 51: `apply`

### `inv_framework/solvers/diffusion/diffusers_compat.py`

- `_require_diffusers` at line 30
- `DiffusersScheduleAdapter` at line 39: `__init__`, `set_timesteps`, `add_noise`, `_alpha_bar_view`, `step`
- `DiffusersUNetWrapper` at line 102: `__init__`, `forward`

### `inv_framework/solvers/diffusion/dps.py`

- `DPSSolver` at line 20: `__init__`, `solve`

### `inv_framework/solvers/diffusion/scheduler.py`

- `NoiseSchedule` at line 21: `set_timesteps`, `add_noise`, `step`, `predict_x0_from_eps`, `_alpha_bar_view`
- `VPSchedule` at line 60: `__init__`, `set_timesteps`, `_alpha_bar_view`, `add_noise`, `step`

### `inv_framework/solvers/dip.py`

- `DIPSolver` at line 17: `__init__`, `solve`

### `inv_framework/solvers/inr.py`

- `INRSolver` at line 16: `__init__`, `solve`

### `inv_framework/utils/metrics.py`

- `psnr` at line 7
- `ssim` at line 13

### `tests/test_nonlinear_operator.py`

- `_SquaredBlur` at line 23: `__init__`, `forward`
- `test_nonlinear_op_does_not_require_adjoint` at line 42
- `test_classical_refuses_nonlinear` at line 51
- `test_dip_recovers_nonlinear` at line 64

### `tests/test_radon_adjoint.py`

- `test_autograd_matches_adjoint` at line 25
- `_disk_phantom` at line 41
- `test_fbp_recovers_disk` at line 51

## 后续执行要求

后续代理或人工在本目录工作时，应先阅读本文件。凡涉及上述 `.py` 文件的任务，默认结论都是：2026-06-08 已存在的实现被冻结，不得被改写、删除或影响；但允许在这些 `.py` 文件中追加新的、彼此隔离的实现内容。若用户要求继续开发，应把开发范围限制在不会改变上述既有 Python 实现基线的地方；如果任务与本冻结规则冲突，必须先向用户确认新的优先级。

后续新增测试应统一保存到：

```text
D:\Pythonprojects\yan_1\invframework\test
```

当前已经存在的 `tests/` 目录及其中冻结清单列出的既有测试不得移动、删除或覆盖；新增测试放入 `test/`，以便与 2026-06-08 既有测试基线区分。

每次对本项目进行文件改动后，都必须在项目根目录的 `report.md` 中记录本次改动（使用中文）；如果 `report.md` 不存在，应先新建。记录内容至少包括：改动日期、改动文件、改动摘要、是否触及冻结清单。
